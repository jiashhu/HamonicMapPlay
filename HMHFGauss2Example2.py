from netgen.meshing import Mesh, MeshPoint, Element1D, FaceDescriptor, Element0D, Element2D
import numpy as np
from netgen.geom2d import *
from ngsolve import *
import datetime
import pytz

def Pos_Transformer(Pos_GF,dim=None):
    if dim is None:
        dim = Pos_GF.dim
    else:
        assert(Pos_GF.dim == dim)
    N = int(len(Pos_GF.vec)/dim)
    coords = Pos_GF.vec.Reshape(N).NumPy().copy()
    return coords.T

def LogTime(timezone='Asia/Shanghai'):
    format = "%Y-%m-%d %H:%M:%S %z"
    a_datetime = datetime.datetime.now(pytz.timezone(timezone))
    datetime_string = a_datetime.strftime(format) + " " + timezone
    return datetime_string

tauval_set = [2**ii for ii in [-6,-7,-8,-9,-10,-11,-12]]
h = 0.015

for tauval in tauval_set:
    tau = Parameter(tauval)
    L = 1/2
    periodic = SplineGeometry()
    pnts = [ (-L,-L), (L,-L), (L,L), (-L,L) ]
    pnums = [periodic.AppendPoint(*p) for p in pnts]

    ldown = periodic.Append ( ["line", pnums[0], pnums[1]],bc="outer")
    lright = periodic.Append ( ["line", pnums[1], pnums[2]], bc="outer")
    periodic.Append ( ["line", pnums[3], pnums[2]], leftdomain=0, rightdomain=1, bc="outer")
    periodic.Append ( ["line", pnums[0], pnums[3]], leftdomain=0, rightdomain=1, bc="outer")
    ngmesh = periodic.GenerateMesh(maxh=h)

    mymesh = Mesh(ngmesh)
    print(mymesh.nv)

    fesD = H1(mymesh, order = 1, dirichlet=mymesh.Boundaries(".*"))
    ## mapping的解空间，映射到3维
    fesVD = fesD**3
    ## Constraint在边界点上不需要设置，因此也不需要multiplier
    fes = H1(mymesh, order = 1, dirichlet=mymesh.Boundaries(".*"))

    ir = IntegrationRule(points = [(0,0), (1,0), (0,1)], weights = [1/6, 1/6, 1/6])
    dx_lumping = dx(intrules = { TRIG : ir })

    n_collo = 2
    fes_ALL = fesVD*fesVD*fes*fes
    U_N = fes_ALL.TrialFunction()
    U0, U1, lamb0, lamb1 = U_N
    V_N = fes_ALL.TestFunction()
    V0, V1, mu0, mu1 = V_N

    U_old = GridFunction(fesVD) ## Set 整点solution 既是解，也为后续外插做准备
    Sol = GridFunction(fes_ALL) 
    U_collo = [] ## Set Collocation 值为后续外插做准备
    U_extr = [] ## Set extrapolation for constraint
    for ii in range(n_collo):
        U_extr.append(GridFunction(fesVD))
        U_collo.append(GridFunction(fesVD))
        
    A_np = np.array([[1/4,1/4-sqrt(3)/6],[1/4+sqrt(3)/6,1/4]])
    b = [1/2,1/2]
    c = [(3-np.sqrt(3))/6, (3+np.sqrt(3))/6]

    Lhs = BilinearForm(fes_ALL)
    Lhs += InnerProduct(U0,V0)*dx + InnerProduct(U1,V1)*dx \
        + tau*(A_np[0,0]*InnerProduct(grad(U0),grad(V0))
            +A_np[0,1]*InnerProduct(grad(U1),grad(V0))
            +A_np[1,0]*InnerProduct(grad(U0),grad(V1))
            +A_np[1,1]*InnerProduct(grad(U1),grad(V1))
            )*dx
    # Lhs += InnerProduct(grad(U0),grad(V0))*dx + InnerProduct(grad(U1),grad(V1))*dx

    Lhs += InnerProduct(U_extr[0],V0)*lamb0*dx_lumping + InnerProduct(U_extr[1],V1)*lamb1*dx_lumping
    Lhs += InnerProduct(U_extr[0],U0)*mu0*dx_lumping + InnerProduct(U_extr[1],U1)*mu1*dx_lumping

    Rhs = LinearForm(fes_ALL)
    Rhs += - InnerProduct(grad(U_old),grad(V0))*dx_lumping - InnerProduct(grad(U_old),grad(V1))*dx

    Eng_set = []
    eps = np.inf
    eps0 = 1e-2
    N_iter = 0

    GetEnergy = lambda expr: 1/2*Integrate(InnerProduct(grad(expr),grad(expr)), mymesh, element_wise=False)
    GetH1 = lambda expr: np.sqrt(Integrate( InnerProduct(grad(expr),grad(expr))+InnerProduct(expr,expr), mymesh, element_wise=False))
    GetL2 = lambda expr: np.sqrt(Integrate( InnerProduct(expr,expr), mymesh, element_wise=False))

    xnorm = Norm(CF((x,y)))
    factor = (xnorm**2 + 1)**(-1)
    uexact = CF((factor*2*x,factor*2*y,factor*(1-xnorm**2)))
    U_old.Interpolate(uexact)
    print('Energy of exact solution is {}'.format(GetEnergy(U_old)))
    
    perturb = uexact + CF(( sin(2*pi*x)*sin(2*pi*y), 
                           sin(2*pi*x)*sin(2*pi*y), 
                           -sin(2*pi*x)*sin(2*pi*y)))
    U_old.Interpolate(perturb)
    ## 压缩初值使其满足归一化条件
    res = Pos_Transformer(U_old)
    res2 = res/np.linalg.norm(res,axis =1)[:,None]
    U_old.vec.data = BaseVector(res2.flatten('F'))
    print('Energy of perturbed initial mapping is {}'.format(GetEnergy(U_old)))

    Variation = GridFunction(fes)
    Variation.Interpolate(InnerProduct(U_old,U_old))
    delta_viration = Integrate(Variation-1,mymesh,element_wise=False)
    print("Initial violation is {}".format(delta_viration))

    ## 通过给出 U_old, U_extr 进行计算，得到满足边界条件的U0和U1（应该是Dirichlet的0边界条件）
    tau_mid = Parameter(c[0]*tauval)
    fes_Euler = fesVD*fes
    U_Euler = fes_Euler.TrialFunction()
    UEuler0, lambEuler0 = U_Euler
    V_Euler = fes_Euler.TestFunction()
    VEuler0, muEuler0 = V_Euler
    Sol_E = GridFunction(fes_Euler)

    Lhs_E = BilinearForm(fes_Euler)
    Lhs_E += InnerProduct(UEuler0,VEuler0)*dx + tau_mid*(InnerProduct(grad(UEuler0),grad(VEuler0)))*dx

    Rhs_E = LinearForm(fes_Euler)
    Rhs_E += - InnerProduct(grad(U_old),grad(VEuler0))*dx
    Lhs_E += InnerProduct(U_old,VEuler0)*lambEuler0*dx_lumping
    Lhs_E += InnerProduct(U_old,UEuler0)*muEuler0*dx_lumping
    Lhs_E.Assemble()
    Rhs_E.Assemble()

    Sol_E.vec.data = Lhs_E.mat.Inverse(inverse="pardiso", freedofs=fes_Euler.FreeDofs())*Rhs_E.vec
    U_extr[0].vec.data = BaseVector(U_old.vec.FV().NumPy() + c[0]*tauval*Sol_E.components[0].vec.FV().NumPy())

    tau_mid.Set(c[1]*tauval)
    Lhs_E.Assemble()
    Sol_E.vec.data = Lhs_E.mat.Inverse(inverse="pardiso", freedofs=fes_Euler.FreeDofs())*Rhs_E.vec
    U_extr[1].vec.data = BaseVector(U_old.vec.FV().NumPy() + c[1]*tauval*Sol_E.components[0].vec.FV().NumPy())
    ## 设定初始的 U_extr: 
    # U_extr[0].vec.data = U_old.vec
    # U_extr[1].vec.data = U_old.vec

    SetNumThreads(40)
    with TaskManager():
        while eps>eps0 and N_iter < 5000:
            ## 每个时间步更新U_old，U_extr（用来计算新的时间区间上的collocation点值，满足正交性条件），再用外插分别更新这两项
            Lhs.Assemble()
            Rhs.Assemble()
            
            Sol.vec.data = Lhs.mat.Inverse(inverse="pardiso", freedofs=fes_ALL.FreeDofs())*Rhs.vec
            
            ## 储存n-1时刻的mapping
            uval = U_old.vec.FV().NumPy()
            ## 计算collocation点上的时间导数
            Sollist = Sol.components
            ## 更新collocation插值点
            for ii in range(n_collo):
                U_collo[ii].vec.data = BaseVector(uval + tauval*(A_np[ii,0]*Sollist[0].vec.FV().NumPy() 
                                                                + A_np[ii,1]*Sollist[1].vec.FV().NumPy()))
            ## 更新解
            U_old.vec.data = BaseVector(uval + tauval*(b[0]*Sollist[0].vec.FV().NumPy() 
                                                    + b[1]*Sollist[1].vec.FV().NumPy()))
            
            ## 利用更新的collocation以及右端点进行外插
            tmpval0 = U_collo[0].vec.FV().NumPy()
            tmpval1 = U_collo[1].vec.FV().NumPy()
            uval = U_old.vec.FV().NumPy()
            U_extr[0].vec.data = BaseVector(2*(3- np.sqrt(3))*uval + (3*np.sqrt(3) - 5)*tmpval0 - np.sqrt(3)*tmpval1)
            U_extr[1].vec.data = BaseVector(2*(3+ np.sqrt(3))*uval + np.sqrt(3)*tmpval0 - (5+3*np.sqrt(3))*tmpval1)

            eps = GetL2(Sollist[0])+GetL2(Sollist[1])
            N_iter = N_iter + 1
            Eng_set.append(GetEnergy(U_old))
            if round(N_iter/10)*10 == N_iter:
                print("{}: Iter {} with Energy {} eps {}"
                      .format(LogTime(),N_iter,Eng_set[-1],eps))


    Variation.Interpolate(InnerProduct(U_old,U_old)-1)
    delta_viration = Integrate(sqrt(Variation**2),mymesh,element_wise=False)
    print("tau = {} and vari = {}".format(-np.log(tauval)/np.log(2),delta_viration))
    print("Final N_iter is {}".format(N_iter))
    print("Final eneryg is {}".format(Eng_set[-1]))
    np.save("Eng_tau{}.npy".format(-np.log(tauval)/np.log(2)),Eng_set)
    vtk = VTKOutput(ma=mymesh,coefs=[U_old],names=["d"],
                    filename="./test_tau{}.vtu".format(-np.log(tauval)/np.log(2)),
                    subdivision=0,legacy=False)
    vtk.Do()
    
    
