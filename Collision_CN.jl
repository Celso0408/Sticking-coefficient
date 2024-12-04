####################################################################################################################################
############################################--- Dynamic of Atomic Collision---######################################################
####################################################################################################################################
# export JULIA_NUM_THREADS=8 			Put this in the terminal to change the number of threads to execute the code
# export  OPENBLAS_NUM_THREADS=8
###################################################################################################################################
####################################################--- Importing ---##############################################################
###################################################################################################################################
using LinearAlgebra								# Linear Algebra Package 
using Base.Threads								#
#BLAS.set_num_threads(1)							# 
####################################################################################################################################
#########################################----Setting the collision parameters----###################################################
####################################################################################################################################
const Lz  = 8.75								# Size of the Box 
const Lb  = 1.25								# Size of the absorption barrier region 
const Nz  = 800									# Number of Z points							
const T_m = 2000								# Final time calculations
const Ni  = 1000								# Number of measurement 
const Nb  = 5									# Exponent of multiplicity
const Ng  = Int(2^Nb)								# Number of time steps for each measure step
const Nt  = Int(Ni*(2^12))							# Number of time steps
const Na  = Int(Nt/(Ng*Ni))							# Measure intervals
const FLAG= "Results_t="							# FLAG Output
const FLAGI="0"									# FLAG Input
###################################################################################################################################
#############################################--- Atomic parameters ---#############################################################
###################################################################################################################################
const sgm = 0.50								# How far the initial distribution is from Z0
const Z0 =  5.75								# Initial position in Bohr's
const M =  290.2								# Atomic Mass [in atomic units]
const K_0 = 0.3									# Initial Kinetic energy in eV
const D   = 4.3 								# Band width in eV.
const k0  = sqrt(2*M*K_0/D)							# Initial Kinetic quasi momentum // Kinetic energy
###################################################################################################################################
###########################################--- Eletronic parameters ---############################################################
###################################################################################################################################
const NeT= 35									# Total Number of Electronic states
const Ne = 1									# Number of Electronic states to work with <= NeT
const NeI= 1									# Initial electronic configuration
###################################################################################################################################
############################################---- Initializing ----#################################################################
###################################################################################################################################
Norma 	= (3.14159265*sgm)^(-0.25)						# Approximately the norm of the wave
dt 	= Float32(T_m/Nt)							# Time step
dz	= Float32(Lz/Nz)							# Space step 
neta 	= Float32(dt/(4*M*(dz^2)))						# Convergence parameter Neta <= 0.1 
Delta2 	= zeros(Float32,(Nz+1)*Ne,(Nz+1)*Ne)					# The Discretization of the second derivative
STICK   = zeros(Float32,Ni,1)
println("Total Number of electronic states: ", Ne )
println("Number of threads= ",Threads.nthreads()," dt = ",dt," dz = ",dz," Neta = ",neta, " Norma ", Norma, " k0 ", k0)

for i=1:1:(Nz+1)						# Build the Matrix of the sec. derivative (3 order) (ok. Working!)
	for j=1:1:Ne
		Delta2[(i-1)*Ne+j,(i-1)*Ne+j] = 5/2																			
		if(i > 1)							
			Delta2[(i-1)*Ne + j,(i-1-1)*Ne + j] = -4/3					
		end
		if(i > 2)							
			Delta2[(i-1)*Ne + j,(i-2-1)*Ne + j] = 1/12		
		end
		if(i > 3)							
			Delta2[(i-1)*Ne + j,(i-3-1)*Ne + j] = -0/90				
		end										
		if(i < Nz + 1)							
			Delta2[(i-1)*Ne + j,(i+1-1)*Ne + j] = -4/3				
		end
		if(i < Nz)							
			Delta2[(i-1)*Ne + j,(i+2-1)*Ne + j] = 1/12			
		end
		if(i < Nz-1)							
			Delta2[(i-1)*Ne + j,(i+3-1)*Ne + j] = -0/90				
		end
	end
end


H_eff 	= zeros(Float32,(Nz+1)*Ne,(Nz+1)*Ne)			# The Electronic Hamiltonian
for z=1:1:(Nz+1)						# Build the Matrix (ok. Working!)
	Projections = zeros(Float64,NeT,NeT)
	file_t = open(FLAGI*"_Projections_"*string(z-1)*".txt")
	for i=1:1:(NeT)*(NeT)
		line 	= (i-1)÷(NeT) + 1
		collum	= i - (line-1)*NeT
		global x= parse(Float64,readline(file_t))
		Projections[line,collum] = x
	end
	close(file_t)
	E_elec = zeros(Float64,NeT,NeT)
	file_t = open(FLAGI*"_Energies_R_"*string(z-1)*".txt")
	for i=1:1:(NeT)
		global x= parse(Float64,readline(file_t))
		E_elec[i,i] = x
	end
	close(file_t)
	H_aux_z = Projections*E_elec*transpose(Projections)
	for i =1:1:Ne
		for j=1:1:Ne
			#k_i = Set_elect[i] 
			#k_j = Set_elect[j]
			global H_eff[(z-1)*Ne + i,(z-1)*Ne + j] = Float32(H_aux_z[i,j])
		end
	end
end

Barrier0 = zeros(Float32,Ne*(Nz+1),Ne*(Nz+1))
N_barrier = Int(round(1*Lb/dz))
for i=0:1:Int(round(N_barrier*Ne))                                                       	
        Barrier0[Ne*(Nz+1)-i,Ne*(Nz+1)-i] =  0.0415*abs(sin(1*pi*(i/(N_barrier*Ne))))	        	#0.042, Lz = 08.75, Lb = 1.25
end
###############################################################################################################################
########################################-----Time Evolution Matrix-----########################################################
###############################################################################################################################
Delta2 = (Symmetric(Delta2))
H_eff = (Symmetric(H_eff))
@time begin		
	U_dt = LinearAlgebra.inv((Symmetric(I+im*neta*Delta2+im*(dt/2)*H_eff + 0.5*dt*Barrier0)))*(Symmetric(I-im*neta*Delta2-im*(dt/2)*H_eff-0.5*dt*Barrier0))
end
Delta2 = 0
H_eff = 0
Barrier0 = 0
##############################################################################################################################
###################################------------------ [U(Na*dt)]------------##################################################
##############################################################################################################################
@time begin
for i=1:1:Nb
	local Aux = 1*U_dt
	mul!(U_dt,Aux,Aux)
end
end
Barrier = zeros(Float32,Ne*(Nz+1),Ne*(Nz+1)) + I
for i=0:1:Int(round(1.125*N_barrier*Ne))								# 
	Barrier[Ne*(Nz+1)-i,Ne*(Nz+1)-i] = exp(-0.00008*(i/(N_barrier*Ne)) )				# 
end
U_dt  = Symmetric(Barrier*U_dt*Barrier)
Barrier = 0

##############################################################################################################################
###################################----- The Atomic distribution Vector-----##################################################
##############################################################################################################################
Wave_func = zeros(Complex{Float32},Ne*(Nz+1),1)						# Wave function			
sum_ = 0										#
for z=1:1:(Nz+1)								  	# Integration by Simpsons Method	
	i = NeI
	k = (z-1)*Ne + i
	global Wave_func[k,1] = Norma*exp(-im*k0*(z-1)*dz)*exp(-((z-1)*dz - Z0)*((z-1)*dz -Z0)/(2*sgm))
	if(z%2==0)
		global sum_ += (2/3)*dz*abs2(Wave_func[k,1])
	end
	if(z%2==1)
		global sum_ += (4/3)*dz*abs2(Wave_func[k,1])
	end
	if(z==1)
		global sum_ += (1/3)*dz*abs2(Wave_func[k,1])
	end
	if(z==Nz+1)
		global sum_ += (1/3)*dz*abs2(Wave_func[k,1])
	end
end
println("k = ",0,"; P(z<L)= ",sum_)
Wave_func = Wave_func/sqrt(sum_)

file_l=open(FLAG*"_Total_"*string(0),"w")
file_s=open(FLAG*string(0),"w")
sum_ = 0
for z=1:1:(Nz+1)
	sum = 0
	for i=1:1:Ne
		k = (z-1)*Ne + i
		write(file_l,string(abs2(Wave_func[k,1])),"\n")
		sum += abs2(Wave_func[k,1])
		local factor_int = 1
		if((z)%(2)==0)
			factor_int = 2/3
		end
		if((z)%(2)==1)
			factor_int = 4/3
		end
		if(z==1)
			factor_int = 1/3
		end
		if(z==Nz + 1)
			factor_int = 1/3
		end
		global sum_ += dz*factor_int*abs2(Wave_func[k,1])
	end
	write(file_s,string(sum),"\n")
end
close(file_s)
close(file_l)

println("k = ",0,"; P(z<L)= ",sum_)
NN = Int(round(100*(Nt/Ng)/T_m))
@time begin
############################################################################################################################
###############################################-----Time Evolution-----#####################################################
############################################################################################################################
k = 1															#
for t=1:1:Int(Nt/Ng)													#
	Wave_func_ = 1*Wave_func + 0*Wave_func										#
	Aux_WF_ = 1*Wave_func + 0*Wave_func										#
	mul!(Wave_func,U_dt,Wave_func_)											# Time evolution of the Wave Function
	if(t == k*Na)													#
		global k +=1												# 
		global Norma = 0											# 
		file=open(FLAG*string(k-1)*"_Total_","w")								#
		file_=open(FLAG*string(k-1),"w")									#
		for z=1:1:(Nz+1)											#
			factor_int = 1											#
			if((z)%(2)==0)											#
				factor_int = 2/3									#
			end												#
			if((z)%(2)==1)											#
				factor_int = 4/3									#
			end												#
			if(z==1)											#
				factor_int = 1/3									#
			end												#
			if(z==Nz + 1)											#
				factor_int = 1/3									#
			end												#
		sum = 0													#
		for i=1:1:Ne												#
			Q = (z-1)*Ne + i										#
			sum += abs2(Wave_func[Q,1])									#
			global Norma += factor_int*dz*(abs2(Wave_func[Q,1]))						# 
			write(file,string(abs2(Wave_func[Q,1])),"	") 						#
			write(file,string(real(Wave_func[Q,1])),"	",string(imag(Wave_func[Q,1])),"	") 	#
			write(file, string(real(Aux_WF_[Q,1])),"	",string(imag(Aux_WF_[Q,1])),"\n") 		#   
		end													#
			write(file_,string(sum),"\n")									#
		end													#
		println("k = ",k-1,"; P(z<L)= ",Norma)									#
		close(file_)												#
		close(file)												#
		STICK[k-1,1]=Norma											#
		if(t <= NN)												#
			global Wave_func = Wave_func/sqrt(Norma) 							#
		elseif(Norma > 1)											#
			global Wave_func = Wave_func/sqrt(Norma) 							#
		end													#
	end														#
end															#
###############################################################################################################################
end

file__=open("STICK","w")                                   		#
for i=1:1:Ni									#
	t = T_m*i/Ni								#
        write(file__,string(t),"	",string(100*STICK[i,1]),"\n")   	#
end										#
close(file__)                                                   		#

