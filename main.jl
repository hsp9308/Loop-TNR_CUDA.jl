using Optim
include("loop_tnr.jl")

br = 1.1199
bi = 0.0
hr = 0.0
Dcut = 70
N = 10 # # of CG steps => # of spins = 2^N
Lp = 2.0^N

r = 2.0/(sqrt(5)+1.0)


x_lower =  0.003171235425658618
x_upper =  0.003171239050983125

###########################################
# Short description about lnZ function : 
#
# Initially, lnZ returns the log of partition function.
# Now. lnZ function returns |Z(h_I)/Z(0)| for a fixed temperature. 
#
# lnZ_h0.dat contains the log of partition function at zero field 
# for the visualization in the Lee-Yang zero searching process.
# However, it can contain arbitrary values if one just want to find the Lee-Yang zeros. 
###########################################

function lnZ(hi) 
  if isfile("lnZ_h0.dat") == false
    init = partition(br,bi,hr,0,Dcut,N,"XY")
    io = open("lnZ_h0.dat","w")
    write(io,string(real(init)),"\n")
    close(io)
  else
    io = open("lnZ_h0.dat","r")
    init = parse(Complex{Float64},readline(io))
    close(io)
  end

  io2 = open("log_f.dat","a+")
  val = partition(br,bi,hr,hi,Dcut,N,"XY")
  print("val=",val,"\n")
  outp = exp((val-init)*Lp)
  print("outp=",outp,"\n")
  write(io2,string(hi),"\t",string(abs(outp)^2),"\n")
  close(io2)
  return abs(outp)^2
end

let
  @time begin
  res = optimize(lnZ,x_lower,x_upper,Brent())
#  res = optimize(lnZ,0.003,0.0032)
  end
  io = open("log.txt", "w")
  write(io,string(res),"\n")
  close(io)
end

