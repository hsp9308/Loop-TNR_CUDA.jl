#=
Implementation of loop-TNR (GPU ver)
with CUDA.jl

by Seongpyo Hong

Date : 2020/12/24
=#

using LinearAlgebra
using Combinatorics
using CUDA
using SpecialFunctions
using TensorOperations

Cplx = Complex{Float64}
EF_iter = 100          # EF maximum iteration number.
EF_error_bound = 1e-15 # Iteration criteion in EF.
sing_bound = 1e-12 # singular value lower bound : discard the value less than this.  
Opt_iter = 200   # Maximum iteration number in loop optimization
# delta : Criterion for the update of the cost function: 
# If change rate of the cost for every 10 iteration is less than delta, iteration stops.
delta = 1e-4
noise = 0.0  # Not used now. 
Dmax = 10000 # Used in the full SVD. Not used now. 


################################
# Order of 4D Array Ta and Tb
#
#       2               4
#       |               |
#  1 ---Ta--- 3    3 ---Tb--- 1
#       |               |
#       4               2
#
################################

function make_Tb(T::Array)
  Tb = permutedims(T,[3,4,1,2])
  return Tb
end



function EF_norm(T::Any)
# Normalize any tensor T with maximum magnitude of the tensor being 1.0.
  max_val = maximum(broadcast(abs,T))
  T = T / max_val
  return T, max_val
end

function lin_solve(N,W)
# Solver for the linear equation N*x = W. 
  Ns = reshape(N,size(N,1)*size(N,2),size(N,3)*size(N,4))
  Ws = reshape(W,size(W,1)*size(W,2),size(W,3))
  x = Ns \ Ws
  x = reshape(x,size(N,1),size(N,2),size(W,3))
  return permutedims(x,(1,3,2))
end

function initial_tensor_ising(br::Real,bi::Real)
  W = zeros(Cplx,2,2)
  beta = complex(br,bi)
  W[1,1] = sqrt(cosh(beta))
  W[1,2] = sqrt(sinh(beta))
  W[2,1] = W[1,1]
  W[2,2] = -W[1,2]

  I = zeros(Cplx,2,2,2,2)
  for i in 1:2
    I[i,i,i,i] = 1.0
  end

  T = zeros(Cplx,2,2,2,2)

  @tensor T[i,j,k,l] = I[a,b,c,d]*W[a,i]*W[b,j]*W[c,k]*W[d,l]

  return T
end

function initial_tensor_XY(br::Real,bi::Real,hr::Real,hi::Real,dcut::Int)

  dc = 80
  bes_b = Complex{Float64}[]
  bes_bh = Complex{Float64}[]
  beta = complex(br,bi)
  h = complex(hr,hi)
  for i in 1:dc-1
    append!(bes_b,besseli(abs(0.5*(dc-2)-i+1),beta))
  end
  for i in 0:2*(dc-1)-2
    append!(bes_bh,besseli(i,beta*h))
  end

  T = zeros(Cplx,dc-1,dc-1,dc-1,dc-1)
  for i in 1:dc-1, j in 1:dc-1, k in 1:dc-1, l in 1:dc-1
    val = sqrt(bes_b[i]*bes_b[j]*bes_b[k]*bes_b[l])*bes_bh[abs(i+j-k-l)+1]
    T[i,j,k,l] = val
  end
  bes_b = []
  bes_bh = []
  return T
end

function EF_stage(Ta, Tb, Dcut)
  if size(Ta,1) == 1 || size(Ta,2) == 1 || size(Ta,3) == 1 || size(Ta,4) == 1
    return Ta, Tb
  end
  PL = Any[0,0,0,0]
  PR = Any[0,0,0,0]

  chi = size(Ta,1)

  print(size(Ta),size(Tb),"\n")
  T = Any[0,0,0,0]
  T[1] = Ta
  T[2] = (permutedims(Tb,[4,1,2,3]))
  T[3] = (permutedims(Ta,[3,4,1,2]))
  T[4] = (permutedims(Tb,[2,3,4,1]))


  # Loop over inner bonds of a block composed of 4 local tensors: To find the projectors.
  for k in 1:4
    baerr = 100
    berr1 = 100
    Tp = Any[0,0,0,0]
    Tp[1] = CuArray(T[k])
    Tp[2] = CuArray(T[k%4+1])
    Tp[3] = CuArray(T[(k+1)%4+1])
    Tp[4] = CuArray(T[(k+2)%4+1])
    L = CuArray(Matrix{Complex{Float64}}(I,size(Tp[1],1),size(Tp[1],1))+noise*rand(Float64,size(Tp[1],1),size(Tp[1],1)))
    temp = CuArray(zeros(Cplx,chi,chi,chi,chi))
    for sweep in 1:EF_iter
      pL = copy(L)
      for s1 in 1:4
	@tensor temp[a,b,c,d] := L[a,i]*Tp[s1][i,b,c,d]
	tmp = reshape(temp,size(temp,1)*size(temp,2)*size(temp,3),size(temp,4))
	_, L = qr(tmp)
	D = CuArray(Diagonal(sign.(diag(L))))
	L = D*L
      end
      L, _ = EF_norm(L)
      err = norm(pL-L)
      aerr = norm(broadcast(abs,real(pL))+broadcast(abs,imag(pL))*im-broadcast(abs,real(L))-broadcast(abs,imag(L))*im)
      if err < EF_error_bound
        break
      elseif aerr < 1e-14
	break
      elseif (abs(aerr-baerr)/baerr)< 1e-4
        break
      end
      print(err,"\t", aerr," A\n")
      berr1 = err
      baerr = aerr
    end

    R = CuArray(Matrix{Complex{Float64}}(I,size(Tp[1],1),size(Tp[1],1))+noise*rand(Float64,size(Tp[1],1),size(Tp[1],1)))
    berr2 = 100
    baerr = 100
    for sweep in 1:EF_iter
      pR = copy(R)
      for s1 in 1:4
        @tensor temp[a,b,c,d] := Tp[5-s1][a,b,c,i]*R[i,d]
	tmp = permutedims(temp,(4,3,2,1))
	tmp = reshape(tmp,size(tmp,1)*size(tmp,2)*size(tmp,3),size(tmp,4))
	_, R = qr(tmp)
	D = CuArray(Diagonal(sign.(diag(R))))
	R = D*R
	R = permutedims(R,(2,1))
      end
      R, _ = EF_norm(R)
      err = norm(pR-R)
      aerr = norm(broadcast(abs,real(pR))+broadcast(abs,imag(pR))*im-broadcast(abs,real(R))-broadcast(abs,imag(R))*im)
      if err < EF_error_bound
        break
      elseif aerr < 1e-14
        break
      elseif (abs(aerr-baerr)/baerr)< 1e-4
        break
      end 

      print(err,"\t", aerr," B\n")
      berr2 = err
      baerr = aerr
    end
    Tp = []
    temp = []
    @tensor LR[a,b] := L[a,i]*R[i,b]
    F = svd(LR)
    U = F.U
    S = F.S
    V = F.Vt
    F = []
    maxd = 1
    for s1 in 1:size(S,1)
      if S[s1] > sing_bound
        maxd = s1
      end
    end
    U = U[:,1:maxd]
    V = V[1:maxd,:]
    S = S[1:maxd]
    S = sqrt.(S)
    S = convert(CuArray{Complex{Float64},1},S)
    Is = zeros(Cplx,maxd,maxd)
    for s1 in 1:maxd
      Is[s1,s1] = 1.0
    end
    Is = CuArray(Is)
    S = Diagonal(S) \ Is
    U = CuArray(conj(transpose(U)))
    V = CuArray(conj(transpose(V)))
    PR[(k+2)%4+1] = CUDA.zeros(Cplx,chi,size(S,1))
    PL[k] = CUDA.zeros(Cplx,size(S,1),chi)
    @tensor PR[(k+2)%4+1][a,b] := R[a,i]*V[i,j]*S[j,b]
    @tensor PL[k][a,b] := S[a,i]*U[i,j]*L[j,b]
  end
  T = []
  Ta = CuArray(Ta)
  Tb = CuArray(Tb)
  #Tna = CUDA.zeros(Cplx,size(PL[1],1),size(PR[3],2),size(PL[3],1),size(PR[1],2))
  @tensor begin
    Tna[a,b,c,d] := PL[1][a,i]*Ta[i,b,c,d]
    Tna[a,b,c,d] := PR[3][j,b]*Tna[a,j,c,d]
    Tna[a,b,c,d] := PL[3][c,k]*Tna[a,b,k,d]
    Tna[a,b,c,d] := PR[1][l,d]*Tna[a,b,c,l]
  end
  Tna = Array(Tna)
  #Tnb = CUDA.zeros(Cplx,size(PR[4],2),size(PL[4],1),size(PR[2],2),size(PL[2],1))
  @tensor begin
    Tnb[a,b,c,d] := PR[4][i,a]*Tb[i,b,c,d]
    Tnb[a,b,c,d] := PL[4][b,j]*Tnb[a,j,c,d]
    Tnb[a,b,c,d] := PR[2][k,c]*Tnb[a,b,k,d]
    Tnb[a,b,c,d] := PL[2][d,l]*Tnb[a,b,c,l]
  end
  Tnb = Array(Tnb)
  return Tna, Tnb
end

function CG_stage(Ta, Tb, Dcut)
  chi = max(size(Ta,1),size(Ta,2),size(Ta,3),size(Ta,4))
  Dn = min(chi*chi,Dcut,400)
  C = Any[0, 0, 0, 0]
  D = copy(C)
  T = copy(C)

  T[1] = CuArray(Ta)
  T[2] = CuArray(permutedims(Tb,(4,1,2,3)))
  T[3] = CuArray(permutedims(Ta,(3,4,1,2)))
  T[4] = CuArray(permutedims(Tb,(2,3,4,1)))
  temp = copy(T[1])
  temp = reshape(temp,size(temp,1)*size(temp,2),size(temp,3)*size(temp,4))
  M = svd(temp)
  S = []
  U = []
  V = []
  S_size = size(M.S,1)
  S = M.S
  trunc_ind = 100
  for i in 1:size(S,1)
    if S[i] > 1e-13
      Dn = i
    end
  end
  if size(T[1],1) == 2 || size(T[1],2) == 2 || size(T[1],3) == 2 || size(T[1],4) == 2
    print(T[1],"\n\n",T[2],"\n\n")
  end
  print("T size = ",size(T[1]),"\n")
  print("U=",size(M.U),"\n")
  print("S=",size(S),"\n")
  print("V=",size(M.Vt),"\n")
  print("Dn=",Dn,"\n")
#  if(size(T[1],1)==1 || size(T[1],2)==1 || size(T[1],3)==1 || size(T[1],4)==1)
#    Dt = 1
#  end

  DDn = min(Dmax,Dn)

  U = M.U[:,1:DDn]
  S = sqrt.(M.S[1:DDn])
  V = M.Vt[1:DDn,:]

  A1 = U*Diagonal(S); A1 = reshape(A1,size(T[1],1),size(T[1],2),size(S,1))
  A2 = Diagonal(S)*V; A2 = reshape(A2,size(S,1),size(T[1],3),size(T[1],4))

  C[1] = copy(A1); D[1] = copy(A2)
  C[3] = copy(A2); C[3] = permutedims(C[3],(2,3,1))
  D[3] = copy(A1); D[3] = permutedims(D[3],(3,1,2))
  
  U = []; V = []; S = []

  temp = copy(T[2])
  temp = reshape(temp,size(temp,1)*size(temp,2),size(temp,3)*size(temp,4))
  M = svd(temp)
  S_size = size(M.S,1)
  S = M.S
  print("T size = ",size(T[2]),"\n")
  print("U=",size(M.U),"\n")
  print("S=",size(S),"\n")
  print("V=",size(M.Vt),"\n")
  for i in 1:size(S,1)
    if S[i] > 1e-13
      Dn = i 
    end 
  end 


  DDn = min(Dmax,Dn)

  U = M.U[:,1:DDn]
  S = sqrt.(M.S[1:DDn])
  V = M.Vt[1:DDn,:]
  A1 = U*Diagonal(S); A1 = reshape(A1,size(T[2],1),size(T[2],2),size(S,1))
  A2 = Diagonal(S)*V; A2 = reshape(A2,size(S,1),size(T[2],3),size(T[2],4))
  C[2] = copy(A1); D[2] = copy(A2)
  C[4] = copy(A2); C[4] = permutedims(C[4],(2,3,1))
  D[4] = copy(A1); D[4] = permutedims(D[4],(3,1,2))

  A1 = []; A2 = []

  C, D = EF_in_CG(C, D, Dcut)

  print("C[1]: ", size(C[1]),"\n")
  print("D[1]: ", size(D[1]),"\n")

  Cd = conj(copy(C))
  Dd = conj(copy(D))

  # Initial error calculation
  # 1. TTp_val
  temp = conj(copy(T[1]))
  @tensor TTp[a,b,c,d] := T[1][a,i,j,c]*temp[b,i,j,d]
  temp = conj(copy(T[2]))
  @tensor begin
    temp2[a,b,c,d] := T[2][a,i,j,c]*temp[b,i,j,d]
    TTp[-1,-2,-3,-4] := TTp[-1,-2,1,2]*temp2[1,2,-3,-4]
  end
  temp = conj(copy(T[3]))
  @tensor begin
    temp2[a,b,c,d] := T[3][a,i,j,c]*temp[b,i,j,d]
    TTp[-1,-2,-3,-4] := TTp[-1,-2,1,2]*temp2[1,2,-3,-4]
  end
  temp = conj(copy(T[4]))
  @tensor begin
    temp2[a,b,c,d] := T[4][a,i,j,c]*temp[b,i,j,d]
    TTp_val[:] := TTp[3,4,1,2]*temp2[1,2,3,4]
  end
  # 2. N_val
  @tensor begin
    N[a,b,c,d] := C[1][a,i,c] * Cd[1][b,i,d]
    temp[a,b,c,d] := D[1][a,i,c] * Dd[1][b,i,d]
    N[-1,-2,-3,-4] := N[-1,-2,1,2] * temp[1,2,-3,-4]
    temp[a,b,c,d] := C[2][a,i,c] * Cd[2][b,i,d]
    N[-1,-2,-3,-4] := N[-1,-2,1,2] * temp[1,2,-3,-4]
    temp[a,b,c,d] := D[2][a,i,c] * Dd[2][b,i,d]
    N[-1,-2,-3,-4] := N[-1,-2,1,2] * temp[1,2,-3,-4]
    temp[a,b,c,d] := C[3][a,i,c] * Cd[3][b,i,d]
    N[-1,-2,-3,-4] := N[-1,-2,1,2] * temp[1,2,-3,-4]
    temp[a,b,c,d] := D[3][a,i,c] * Dd[3][b,i,d]
    N[-1,-2,-3,-4] := N[-1,-2,1,2] * temp[1,2,-3,-4]
    temp[a,b,c,d] := C[4][a,i,c] * Cd[4][b,i,d]
    N[-1,-2,-3,-4] := N[-1,-2,1,2] * temp[1,2,-3,-4]
    temp[a,b,c,d] := D[4][a,i,c] * Dd[4][b,i,d]
    N_val[:] := N[3,4,1,2]*temp[1,2,3,4]
  end
  temp = []
  N = []
  # 3. W_val
  @tensor begin
    temp[a,b,c,d] := Cd[1][a,b,i]*Dd[1][i,c,d]
    W[a,b,c,d] := T[1][a,i,j,c]*temp[b,i,j,d]
    temp[a,b,c,d] := Cd[2][a,b,i]*Dd[2][i,c,d]
    temp[a,b,c,d] := T[2][a,i,j,c]*temp[b,i,j,d]
    W[a,b,c,d] := W[a,b,i,j]*temp[i,j,c,d]
    temp[a,b,c,d] := Cd[3][a,b,i]*Dd[3][i,c,d]
    temp[a,b,c,d] := T[3][a,i,j,c]*temp[b,i,j,d]
    W[a,b,c,d] := W[a,b,i,j]*temp[i,j,c,d]
    temp[a,b,c,d] := Cd[4][a,b,i]*Dd[4][i,c,d]
    temp[a,b,c,d] := T[4][a,i,j,c]*temp[b,i,j,d]
    W_val[:] := W[3,4,1,2] * temp[1,2,3,4]
  end
  temp = []
  W = []
  err = TTp_val[1] + N_val[1] - W_val[1] - conj(W_val[1])
  print("Initial norm = ",TTp_val[1],"\n")
  print("initial error = ",err,"\n")
  berr = err
  berr10 = err
  br_ind = 0


  for sweep in 1:Opt_iter
    if abs(err) < 1e-15
       break
    end
	
    for k in 1:4 # Forward
      k1 = k%4 + 1
      k2 = (k+1)%4 + 1
      k3 = (k+2)%4 + 1
      # 1. making N
      @tensor begin 
        N_tmp[a,b,c,d] := C[k1][a,i,c]*Cd[k1][b,i,d]
        temp[a,b,c,d] := D[k1][a,i,c]*Dd[k1][b,i,d]
        N_tmp[a,b,c,d] := N_tmp[a,b,i,j]*temp[i,j,c,d]
        temp[a,b,c,d] := C[k2][a,i,c]*Cd[k2][b,i,d]
        N_tmp[a,b,c,d] := N_tmp[a,b,i,j]*temp[i,j,c,d]
        temp[a,b,c,d] := D[k2][a,i,c]*Dd[k2][b,i,d]
        N_tmp[a,b,c,d] := N_tmp[a,b,i,j]*temp[i,j,c,d]
        temp[a,b,c,d] := C[k3][a,i,c]*Cd[k3][b,i,d]
        N_tmp[a,b,c,d] := N_tmp[a,b,i,j]*temp[i,j,c,d]
        temp[a,b,c,d] := D[k3][a,i,c]*Dd[k3][b,i,d]
        N_tmp[a,b,c,d] := N_tmp[a,b,i,j]*temp[i,j,c,d]
      end
      @tensor begin
        temp[a,b,c,d] := D[k][a,i,c]*Dd[k][b,i,d]
        N[a,b,c,d] := N_tmp[i,j,c,a]*temp[d,b,i,j]
      end
      # 2. making W
      @tensor begin
        temp[a,b,c,d] := Cd[k1][a,b,i]*Dd[k1][i,c,d]
	W_tmp[a,b,c,d] := T[k1][a,i,j,c]*temp[b,i,j,d]
        temp[a,b,c,d] := Cd[k2][a,b,i]*Dd[k2][i,c,d]
        temp[a,b,c,d] := T[k2][a,i,j,c]*temp[b,i,j,d]
        W_tmp[a,b,c,d] := W_tmp[a,b,i,j]*temp[i,j,c,d]
        temp[a,b,c,d] := Cd[k3][a,b,i]*Dd[k3][i,c,d]
        temp[a,b,c,d] := T[k3][a,i,j,c]*temp[b,i,j,d]
        W_tmp[a,b,c,d] := W_tmp[a,b,i,j]*temp[i,j,c,d]
      end
      @tensor begin
        W_tmp[a,b,c,d] := W_tmp[j,b,i,a]*T[k][i,c,d,j]
	W[a,b,c] := W_tmp[a,i,c,j]*Dd[k][b,j,i]
      end
      temp = []

      Cout = lin_solve(N,W)
      Coutd = conj(copy(Cout))
      @tensor CCp_t[a,b,c,d] := Cout[a,i,c]*Coutd[b,i,d]
      @tensor begin
        N_val[:] := N[-1,-2,-3,-4] * CCp_t[-3,-1,-4,-2]
	W_val[:] := W[-1,-2,-3] * Coutd[-1,-3,-2]
      end
      err = TTp_val[1] + N_val[1] - W_val[1] - conj(W_val[1])
      if abs(err) > abs(berr)
        err = berr
      elseif real(err) < 0
	err = berr
      else
        C[k] = copy(Cout); Cd[k] = copy(Coutd); Cout = []; Coutd = []
 	berr = err
      end

      @tensor N[a,b,c,d] := N_tmp[d,b,i,j]*CCp_t[i,j,c,a]
      CCP_t = []
      @tensor W[a,b,c] := W_tmp[i,b,j,c] * Cd[k][i,j,a]
      Dout = lin_solve(N,W)
      Doutd = conj(copy(Dout))
      @tensor DDp_t[a,b,c,d] := Dout[a,i,c]*Doutd[b,i,d]
      @tensor begin
        N_val[:] := N[-1,-2,-3,-4] * DDp_t[-3,-1,-4,-2]
        W_val[:] := W[-1,-2,-3] * Doutd[-1,-3,-2]
      end
      err = TTp_val[1] + N_val[1] - W_val[1] - conj(W_val[1])
      DDp_t = []
      if abs(err) > abs(berr)
        err = berr
      elseif real(err)<0
        err = berr
      else
        D[k] = copy(Dout); Dd[k] = copy(Doutd); Dout = []; Doutd = []
 	berr = err
      end
    end # End of forward optimization

    for k in 4:-1:2 # Backward Optimization
      k1 = k%4 + 1
      k2 = (k+1)%4 + 1
      k3 = (k+2)%4 + 1
      # 1. Making N : Starting from D[k], to C[k3]
      # To optimize in 1 loop : C[k], D[k3]
      @tensor begin
        N_tmp[a,b,c,d] := D[k][a,i,c]*Dd[k][b,i,d]
        temp[a,b,c,d] := C[k1][a,i,c]*Cd[k1][b,i,d]
	N_tmp[a,b,c,d] := N_tmp[a,b,i,j]*temp[i,j,c,d]
        temp[a,b,c,d] := D[k1][a,i,c]*Dd[k1][b,i,d]
        N_tmp[a,b,c,d] := N_tmp[a,b,i,j]*temp[i,j,c,d]
        temp[a,b,c,d] := C[k2][a,i,c]*Cd[k2][b,i,d]
        N_tmp[a,b,c,d] := N_tmp[a,b,i,j]*temp[i,j,c,d]
        temp[a,b,c,d] := D[k2][a,i,c]*Dd[k2][b,i,d]
        N_tmp[a,b,c,d] := N_tmp[a,b,i,j]*temp[i,j,c,d]
        temp[a,b,c,d] := C[k3][a,i,c]*Cd[k3][b,i,d]
        N_tmp[a,b,c,d] := N_tmp[a,b,i,j]*temp[i,j,c,d]
      end
      @tensor begin
        temp[a,b,c,d] := D[k3][a,i,c]*Dd[k3][b,i,d]
        N[a,b,c,d] := N_tmp[d,b,i,j]*temp[i,j,c,a]
      end
      # 2. Making W
      @tensor begin
        temp[a,b,c,d] := Cd[k1][a,b,i]*Dd[k1][i,c,d]
        W_tmp[a,b,c,d] := T[k1][a,i,j,c]*temp[b,i,j,d]
        temp[a,b,c,d] := Cd[k2][a,b,i]*Dd[k2][i,c,d]
        temp[a,b,c,d] := T[k2][a,i,j,c]*temp[b,i,j,d]
        W_tmp[a,b,c,d] := W_tmp[a,b,i,j]*temp[i,j,c,d]
        temp[a,b,c,d] := Cd[k3][a,b,i]*Dd[k3][i,c,d]
        temp[a,b,c,d] := T[k3][a,i,j,c]*temp[b,i,j,d]
        W[a,b,c,d] := W_tmp[a,b,i,j]*temp[i,j,c,d]
        W[a,b,c,d] := W[j,b,i,a]*T[k][i,c,d,j]
	W[a,b,c] := W[a,i,c,j]*Dd[k][b,j,i]
      end
      temp = []
      Cout = lin_solve(N,W)
      Coutd = conj(copy(Cout))
      @tensor CCp_t[a,b,c,d] := Cout[a,i,c]*Coutd[b,i,d]
      @tensor begin
        N_val[:] := N[-1,-2,-3,-4] * CCp_t[-3,-1,-4,-2]
        W_val[:] := W[-1,-2,-3] * Coutd[-1,-3,-2]
      end
      err = TTp_val[1] + N_val[1] - W_val[1] - conj(W_val[1])
      if abs(err) > abs(berr)
        err = berr
      elseif real(err) < 0
        err = berr
      else
        C[k] = copy(Cout); Cd[k] = copy(Coutd); Cout = []; Coutd = []
        berr = err
      end
      
      @tensor  N[a,b,c,d] := N_tmp[i,j,c,a]*CCp_t[d,b,i,j]
      CCp_t = []

      @tensor begin
        temp[a,b,c,d] := Cd[k][a,b,i]*Dd[k][i,c,d]
        temp[a,b,c,d] := T[k][a,i,j,c]*temp[b,i,j,d]
	W[a,b,c,d] := temp[a,b,i,j]*W_tmp[i,j,c,d]
        W[a,b,c,d] := W[j,b,i,a]*T[k3][i,c,d,j]
        W[a,b,c] := W[i,b,j,c] * Cd[k3][i,j,a]
      end
      Dout = lin_solve(N,W)
      Doutd = conj(copy(Dout))
      @tensor DDp_t[a,b,c,d] := Dout[a,i,c]*Doutd[b,i,d]
      @tensor begin
        N_val[:] := N[-1,-2,-3,-4] * DDp_t[-3,-1,-4,-2]
        W_val[:] := W[-1,-2,-3] * Doutd[-1,-3,-2]
      end
      err = TTp_val[1] + N_val[1] - W_val[1] - conj(W_val[1])
      DDp_t = []
      if abs(err) > abs(berr)
        err = berr
      elseif real(err) <0
        err = berr
      else
        D[k3] = copy(Dout); Dd[k3] = copy(Doutd); Dout = []; Doutd = []
        berr = err
      end
    if sweep%10 == 5
      C, D = EF_in_CG(C, D, Dcut)
      Cd = conj(copy(C))
      Dd = conj(copy(D))
    end
    end # End of Backward optimization


    if sweep%10 == 0
       print("# ",sweep,"th iteration, error = \n",sweep,"\t",real(err),"\n")
       if abs(real((berr10-err)/berr10)) < delta
          br_ind = 1
       end
       berr10 = err
       C, D = EF_in_CG(C, D, Dcut)
       Cd = conj(copy(C))
       Dd = conj(copy(D))
    end
    if br_ind == 1
       break
    end

  end # END of iteration loop

  print("Final error = ", err, "\n")
  Cd = []
  Dd = []
  @tensor begin
    Ta[-1,-2,-3,-4] := D[4][-1,1,2]*C[3][3,1,-2]*D[2][-3,4,3]*C[1][2,4,-4]
    Tb[-1,-2,-3,-4] := C[4][1,2,-1]*D[3][-2,3,1]*C[2][4,3,-3]*D[1][-4,2,4]
  end
  return Array(Ta), Array(Tb)
end

function EF_in_CG(C, D, Dcut)
  if size(C[1],1) == 1 || size(C[1],2) == 1 || size(C[1],3) == 1 
    return C, D
  elseif size(D[1],1) == 1 || size(D[1],2) == 1 || size(D[1],3) == 1
    return C, D
  elseif size(C[2],1) == 1 || size(C[2],2) == 1 || size(C[2],3) == 1
    return C, D
  elseif size(D[2],1) == 1 || size(D[2],2) == 1 || size(D[2],3) == 1
    return C, D
  end

  PL = Any[0,0,0,0,0,0,0,0]
  PR = Any[0,0,0,0,0,0,0,0]

  for k in 1:4 # One k values : btw D[k-1] and C[k],  btw C[k] and D[k]
    berr1 = 100
    baerr = 100
    L = CuArray(Matrix{Complex{Float64}}(I,size(C[k],1),size(C[k],1))+noise*rand(Float64,size(C[k],1),size(C[k],1)))
    temp = CuArray(zeros(Cplx,size(C[k],1)))
    for sweep in 1:EF_iter
      pL = copy(L)
      for s1 in 0:3
        @tensor temp[a,b,c] := L[a,i]*C[(k+s1-1)%4+1][i,b,c]
        tmp = reshape(temp,size(temp,1)*size(temp,2),size(temp,3))
        _, L = qr(tmp)
        F = CuArray(Diagonal(sign.(diag(L))))
        L = F*L
        @tensor temp[a,b,c] := L[a,i]*D[(k+s1-1)%4+1][i,b,c]
        tmp = reshape(temp,size(temp,1)*size(temp,2),size(temp,3))
        _, L = qr(tmp)      
	F = CuArray(Diagonal(sign.(diag(L))))
        L = F*L
      end
      L, _ = EF_norm(L)
      err = norm(pL-L)
      aerr = norm(broadcast(abs,real(pL))+broadcast(abs,imag(pL))*im-broadcast(abs,real(L))-broadcast(abs,imag(L))*im)
      if err < EF_error_bound
        break
      elseif aerr < 1e-14
        break
      elseif (abs(aerr-baerr)/baerr)< 1e-4
        break
      end 

#      print(err,"\t",aerr," A\n")
      berr1 = err
      baerr = aerr
    end

    R = CuArray(Matrix{Complex{Float64}}(I,size(C[k],1),size(C[k],1))+noise*rand(Float64,size(C[k],1),size(C[k],1)))
    berr2 = 100
    baerr = 100
    for sweep in 1:EF_iter
      pR = copy(R)
      for s1 in 4:-1:1
        @tensor temp[a,b,c] := D[(5+s1+k+1)%4+1][a,b,i]*R[i,c]
        tmp = permutedims(temp,(3,2,1))
	tmp = reshape(tmp,size(tmp,1)*size(tmp,2),size(tmp,3))
        _, R = qr(tmp)
        F = CuArray(Diagonal(sign.(diag(R))))
        R = F*R
        R = permutedims(R,(2,1))
        @tensor temp[a,b,c] := C[(5+s1+k+1)%4+1][a,b,i]*R[i,c]
        tmp = permutedims(temp,(3,2,1))
	tmp = reshape(tmp,size(tmp,1)*size(tmp,2),size(tmp,3))
        _, R = qr(tmp)
        F = CuArray(Diagonal(sign.(diag(R))))
        R = F*R
        R = permutedims(R,(2,1))
      end
      R, _ = EF_norm(R)
      err = norm(pR-R)
      aerr = norm(broadcast(abs,real(pR))+broadcast(abs,imag(pR))*im-broadcast(abs,real(R))-broadcast(abs,imag(R))*im)
      if err < EF_error_bound
        break
      elseif aerr < 1e-14
        break
      elseif (abs(aerr-baerr)/baerr)< 1e-4
        break
      end 

#      print(err,"\t",aerr," B\n")
      berr2 = err
      baerr = aerr
    end
    @tensor LR[a,b] := L[a,i]*R[i,b]
    F = svd(LR)
    U = F.U
    S = F.S
    V = F.Vt
    maxd = 1
    for s1 in 1:size(S,1)
      if S[s1] > sing_bound && maxd < Dcut
        maxd = s1
      end
    end
    U = U[:,1:maxd]
    V = V[1:maxd,:]
    S = S[1:maxd]
    S = sqrt.(S)
    S = convert(CuArray{Complex{Float64},1},S)
    Is = zeros(Cplx,maxd,maxd)
    for s1 in 1:maxd
      Is[s1,s1] = 1.0
    end
    Is = CuArray(Is)
    S = Diagonal(S) \ Is
    U = CuArray(conj(transpose(U)))
    V = CuArray(conj(transpose(V)))
    PR[(2*k+5)%8+1] = CUDA.zeros(Cplx,size(U,1),size(S,1))
    PL[2*k-1] = CUDA.zeros(Cplx,size(S,1),size(U,1))
    @tensor PR[(2*k+5)%8+1][a,b] := R[a,i]*V[i,j]*S[j,b]
    @tensor PL[2*k-1][a,b] := S[a,i]*U[i,j]*L[j,b]

    L = CuArray(Matrix{Complex{Float64}}(I,size(D[k],1),size(D[k],1))+noise*rand(Float64,size(D[k],1),size(D[k],1)))
    baerr = 100
    for sweep in 1:EF_iter
      pL = copy(L)
      for s1 in 0:3
        @tensor temp[a,b,c] := L[a,i]*D[(k+s1-1)%4+1][i,b,c]
        tmp = reshape(temp,size(temp,1)*size(temp,2),size(temp,3))
        _, L = qr(tmp)
	F = CuArray(Diagonal(sign.(diag(L))))
        L = F*L
        @tensor temp[a,b,c] := L[a,i]*C[(k+s1)%4+1][i,b,c]
        tmp = reshape(temp,size(temp,1)*size(temp,2),size(temp,3))
        _, L = qr(tmp)
	F = CuArray(Diagonal(sign.(diag(L))))
        L = F*L
      end
      L, _ = EF_norm(L)
      err = norm(pL-L)
      aerr = norm(broadcast(abs,real(pL))+broadcast(abs,imag(pL))*im-broadcast(abs,real(L))-broadcast(abs,imag(L))*im)
      if err < EF_error_bound
        break
      elseif aerr < 1e-14
        break
      elseif (abs(aerr-baerr)/baerr)< 1e-4
        break
      end

#      print(err,"\t",aerr," A\n")
      berr1 = err
      baerr = aerr
    end

    R = CuArray(Matrix{Complex{Float64}}(I,size(D[k],1),size(D[k],1))+noise*rand(Float64,size(D[k],1),size(D[k],1)))
    baerr = 100
    for sweep in 1:EF_iter
      pR = copy(R)
      for s1 in 4:-1:1
        @tensor temp[a,b,c] := C[(5+s1+k+2)%4+1][a,b,i]*R[i,c]
        tmp = permutedims(temp,(3,2,1))
	tmp = reshape(tmp,size(tmp,1)*size(tmp,2),size(tmp,3))
        _, R = qr(tmp)
	F = CuArray(Diagonal(sign.(diag(R))))
        R = F*R
        R = permutedims(R,(2,1))
        @tensor temp[a,b,c] := D[(5+s1+k+1)%4+1][a,b,i]*R[i,c]
        tmp = permutedims(temp,(3,2,1))
	tmp = reshape(tmp,size(tmp,1)*size(tmp,2),size(tmp,3))
        _, R = qr(tmp)
	F = CuArray(Diagonal(sign.(diag(R))))
        R = F*R
        R = permutedims(R,(2,1))
      end
      R, _ = EF_norm(R)
      err = norm(pR-R)
      aerr = norm(broadcast(abs,real(pR))+broadcast(abs,imag(pR))*im-broadcast(abs,real(R))-broadcast(abs,imag(R))*im)
      if err < EF_error_bound
        break
      elseif aerr < 1e-14
        break
      elseif (abs(aerr-baerr)/baerr)< 1e-4
        break
      end 
#      print(err,"\t",aerr," B\n")
      berr2 = err
      baerr = aerr
    end
    @tensor LR[a,b] := L[a,i]*R[i,b]
    F = svd(LR)
    U = F.U
    S = F.S
    V = F.Vt
    maxd = 1
    for s1 in 1:size(S,1)
      if S[s1] > sing_bound && maxd < Dcut
        maxd = s1
      end
    end
    U = U[:,1:maxd]
    V = V[1:maxd,:]
    S = S[1:maxd]
    S = sqrt.(S)
    S = convert(CuArray{Complex{Float64},1},S)
    Is = zeros(Cplx,maxd,maxd)
    for s1 in 1:maxd
      Is[s1,s1] = 1.0
    end
    Is = CuArray(Is)
    S = Diagonal(S) \ Is
    U = CuArray(conj(transpose(U)))
    V = CuArray(conj(transpose(V)))
    PR[2*k-1] = CUDA.zeros(Cplx,size(U,1),size(S,1))
    PL[2*k] = CUDA.zeros(Cplx,size(S,1),size(U,1))
    @tensor PR[2*k-1][a,b] := R[a,i]*V[i,j]*S[j,b]
    @tensor PL[2*k][a,b] := S[a,i]*U[i,j]*L[j,b]
    @tensor IDen[a,b] := PR[2*k-1][a,i]*PL[2*k][i,b]
 #   print(size(IDen),"\n")
  end

  for k in 1:4
    @tensor begin
      C[k][a,b,c] := PL[2*k-1][a,i]*C[k][i,b,c]
      C[k][a,b,c] := PR[2*k-1][i,c]*C[k][a,b,i]
      D[k][a,b,c] := PL[2*k][a,i]*D[k][i,b,c]
      D[k][a,b,c] := PR[2*k][i,c]*D[k][a,b,i]
    end
  end

  return C, D
end

function partition(br::Real,bi::Real,hr::Real,hi::Real,Dcut::Int,n::Int,model::String)
  f = 1.0 
  if model == "ising"
    Ta = initial_tensor_ising(br,bi)
  elseif model == "XY"
    Ta = initial_tensor_XY(br,bi,hr,hi,Dcut)
  end

  io = open("lnZ.out", "w")

  Tb = make_Tb(Ta)
  Ta, na = EF_norm(Ta)
  Tb, nb = EF_norm(Tb)
  lnZ = (log(na)+log(nb))*0.5
  for i in 1:n-1
    print(i,"th iteration starts!\n")
    Ta, Tb = EF_stage(Ta,Tb,Dcut)
    Ta, Tb = CG_stage(Ta,Tb,Dcut)

    Ta, na = EF_norm(Ta)
    Tb, nb = EF_norm(Tb)

    f = f*0.5

    lnZ = lnZ + (log(na)+log(nb))*0.5*f
    @tensor val[:] := Ta[-1,-2,-3,-4]*Tb[-1,-2,-3,-4]
    lnZk = lnZ + log(val[1])*0.5*f
    write(io,string(i+1),"\t", string(real(lnZk)),"\t",string(imag(lnZk)), "\n")
  end


  @tensor val[:] := Ta[-1,-2,-3,-4]*Tb[-1,-2,-3,-4]

  lnZ = lnZ + log(val[1])*0.5*f

  write(io,string(n),"\t", string(real(lnZ)),"\t",string(imag(lnZ)), "\n")

  close(io)
  return lnZ
end



