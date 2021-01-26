import numpy as np
import numpy.linalg as la
from scipy import optimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
def solve_disp_eqn(xs,ws,lmbda):
    """Construct integral equation and find where it has a solution
    """
    eqtn = np.zeros((len(xs),len(xs)))
    # Compute the kernel of integral equation
    def IE_kernel(k_,kk_,lambda_,p_):
        """Kernel of integral equation is integral itself
        """
        res = 1./np.pi/(k_**2+kk_**2+p_**2)
        def kern_fun(kkk_):
            """Function to integrate
            """
            ans = 1./(k_**2+(k_-kkk_)**2+p_**2)/(kk_**2+(kk_-kkk_)**2+p_**2)
            sqrt_ = np.sqrt(2.*p_**2+kkk_**2)
            ans *= sqrt_/(sqrt_+lambda_)
            return ans
        ks = (1.+xs)/(1.-xs)
        dkdx = 2./(1.-xs)**2
        integr = sum(ws*kern_fun( ks )* dkdx) + sum(ws*kern_fun( -ks )* dkdx)
        res -= 2.*lambda_/np.pi**2 * integr
        return res
    # Define the matrix of integral equation
    def det_eqtn(p):
        def phi(k_):
            return 1.-1./(k_*k_+p*p)**0.5
        for i in range(len(xs)):
            ki = (1.+xs[i])/(1.-xs[i])
            for j in range(len(xs)):
                kj = (1.+xs[j])/(1.-xs[j])
                dkdx = 2./(1.-xs[j])**2
                eqtn[i,j] = ws[j]*IE_kernel(ki,kj,lmbda,p)*dkdx
                eqtn[i,j] += ws[j]*IE_kernel(ki,-kj,lmbda,p)*dkdx
            eqtn[i,i] -= phi(ki)
        return la.det(eqtn)
        
    return optimize.brenth(det_eqtn,1.,np.sqrt(2))


# precompute gaussian points and weights to use in 
# gaussian-legendre quadratures




def find_F(lmb,x,w):
    """
    Нахождение функции F(k)
    """
    
    p = solve_disp_eqn(x,w,lmb)
    eqtn = np.zeros((len(x),len(x)))
    # Compute the kernel of integral equation
    def IE_kernel(k_,kk_,lambda_,p_):
        """Kernel of integral equation is integral itself
        """
        res = 1./np.pi/(k_**2+kk_**2+p_**2)
        def kern_fun(kkk_):
            """Function to integrate
            """
            ans = 1./(k_**2+(k_-kkk_)**2+p_**2)/(kk_**2+(kk_-kkk_)**2+p_**2)
            sqrt_ = np.sqrt(2.*p_**2+kkk_**2)
            ans *= sqrt_/(sqrt_+lambda_)
            return ans
        ks = (1.+x)/(1.-x)
        dkdx = 2./(1.-x)**2
        integr = sum(w*kern_fun( ks )* dkdx) + sum(w*kern_fun( -ks )* dkdx)
        res -= 2.*lambda_/np.pi**2 * integr
        return res
    
    def phi(k_):
        return 1.-1./(k_*k_+p*p)**0.5
    
    for i in range(len(x)):
        ki = (1.+x[i])/(1.-x[i])
        for j in range(len(x)):
            kj = (1.+x[j])/(1.-x[j])
            dkdxj = 2./(1.-x[j])**2
            eqtn[i,j] = w[j]*IE_kernel(ki,kj,lmb,p)*dkdxj
            eqtn[i,j] += w[j]*IE_kernel(ki,-kj,lmb,p)*dkdxj
        eqtn[i,i] -= phi(ki)
    #решаем задачу на с.з. и выбираем с.в. с наименьшим с.з в качестве решения
    V,F = la.eig(eqtn)
    minposition= V.tolist().index(min(V))
    F = F[:,minposition]
    return F[0:(len(F)-1)],p


def find_G(lmb):
    """
    Ищет Фурье образ волновой функции G(kx,ky)
    """
    pol = 30
    x, w = np.polynomial.legendre.leggauss(pol)
    F,p = find_F(lmb,x,w)
    x = x[0:len(x)-1]
    w = w[0:len(w)-1]
    def funct_H(k):
        def kern_fun(k_):
            
            ans = 2/np.pi*1/((k-k_)**2+k_**2+p**2)
            return ans
        ks = (1.+x)/(1.-x)
        dkdx = 2./(1.-x)**2
        integr = sum(w*kern_fun( ks )*F* dkdx) + sum(w*kern_fun( -ks )*F* dkdx)
        sqrt = 1 + lmb/np.sqrt(k**2 + 2*p**2)
        Hk = integr/sqrt
        return Hk
    H = np.empty_like(F)
    for i in range(len(x)):
        ki = (1.+x[i])/(1.-x[i])
        H[i] = funct_H(ki)
    #Интерполяция  H и F
    kr = (1.+x)/(1.-x)
    interp = np.linspace(kr[0], kr[pol-2], 500)
    iF = np.interp(interp,kr,F)
    iH = np.interp(interp,kr,H)
    plt.figure(1)
    plt.plot(interp, iF ,'x', color = 'b')
    plt.plot(kr, F ,'x', color = 'r')
    plt.title('Функция F')
    plt.figure(2)
    plt.plot(interp, iH , 'x', color = 'b')
    plt.plot(kr, H ,'x', color = 'r')
    plt.title('Функция H')
    #plt.show()
    #H <<F и ей можно пренебречь в обратном преобразовании Фурье
    G1 = np.zeros((len(interp),len(interp)))
    for i in range (len(interp)):
        for k in range (len(interp)):
            G1[k,i] = (iF[i] + iF[k])/(0.5*(interp[i]**2+interp[k]**2+p**2))
    #Функция F(k) найдена только для k>0,и она симметрична, поэтому надо сделать отражения для построения полной G        
    G = np.block([[np.flipud(np.fliplr(G1)),np.flipud(G1)],
                  [np.fliplr(G1),G1]
        ])
    d = interp[1] - interp[0]
    N_intrp = 2*len(interp)
    find_PSI_XY(G,d,N_intrp)
    
def find_PSI_XY(G,d,N):
    """
    Делает обратное преобразование Фурье
    """
    #сдвиг "частот"
    G1 = np.fft.fftshift(G)
    # выполняем IFFT
    IDF2 = np.fft.ifft2(G1)
    #Сдвиг отрицательной "координатной" части влево
    IDF2 = np.fft.fftshift(IDF2)
    # переходим к соответствующему математическому обратному образу Фурье
    MIDF = N**2*d**2*IDF2/(2*np.pi)**2
    #выводим соответствующие "координаты из преобразования" 
    x1 = np.fft.fftfreq(N,d)
    x2 = np.fft.fftfreq(N,d)
    #сдвигаем отрицательные частоты влево
    x1 = np.fft.fftshift(x1)
    x2 = np.fft.fftshift(x2)
    #Переходим к "реальным" координатам
    x1 = 2*np.pi*x1
    x2 = 2*np.pi*x2
    #Строим график в.ф.
    fig = plt.figure(3)
    ax = fig.add_subplot(111, projection='3d')
    SMIDF = MIDF[480:N-480,480:N-480]
    x1 = x1[480:N-480]
    x2 = x2[480:N-480]
    x1, x2 = np.meshgrid(x1, x2)
    ax.plot_surface( x1,x2, SMIDF.real,color = 'r')
    plt.title("Волновая функций PSI(x,y)")
    #plt.show()



#Находит G и обратное преобразование Фурье от него.    
find_G(1.2)
# precompute gaussian points and weights to use in 
# gaussian-legendre quadratures
polyorder = 10
X, W = np.polynomial.legendre.leggauss(polyorder)
N = 30
#Дмапазон именения lambda
ll = np.linspace(0.1,2.6,N)
E = np.empty_like(ll)
for k in range(N):
    E[k] = -1/2*solve_disp_eqn(X,W,ll[k])**2
plt.figure(4)
plt.plot(ll, E ,'x', color = 'r')
plt.xlabel("lambda")
plt.ylabel("E")
plt.title("Зависимость энергии уровня E(lambda)")
plt.show()












    

