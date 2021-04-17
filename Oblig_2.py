## Compute the watted surface
import numpy as np
import matplotlib.pyplot as plt


def Watted_surface(L,D,N1,N2):
    dy = D/N2
    dx = L/N1
    # left and right side
    xm = np.ones(N2+1)*(-L/2)
    xp = np.ones(N2+1)*(L/2)
    coor_left = np.zeros((2,N2+1))
    coor_right = np.zeros((2,N2+1))
    coor_left[0,:] = xm
    coor_right[0,:] = xp
    for i in range(len(coor_left[1])):
        coor_left[1,i] = -dy*i
        coor_right[1,i] = -dy*i

    #bottom side
    M = N1
    yb = -D*np.ones(M-1)
    xb = []
    coor_bot = np.zeros((2,M-1))
    coor_bot[1,:] = yb

    for i in range(1,M):

        x = -L/2 + dx*i
        xb.append(x)

    coor_bot[0,:] = xb

    # put together
    x = np.concatenate([coor_left[0,:],coor_bot[0,:], coor_right[0,:]])
    y = np.concatenate([coor_left[1,:],yb,np.flip(coor_right[1,:])])

    return x,y


D = 1
N2 = 10
L = np.array([2*D,1*D,0.1*D])
N1 = np.array([N2,N2,5])


coord = []

# Plot the geometry
for i in range(len(L)):
    x,y = Watted_surface(L[i],D,N1[i],N2)
    coor = [x,y]
    coord.append(coor)

    plt.plot(x,y,'k',alpha=0.8,linewidth=1.4)
    plt.plot(x,y,'r.',linewidth=7.0)
    plt.title(r'Discretisation surface $S_b$ D = 1 and L = %g'%L[i])
    plt.grid(True)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    if i==2:
        plt.xlim([-0.3, 0.3])
        plt.ylim([-1.02, 0.02])
    else:
        plt.xlim([-1.02, 1.02])
        plt.ylim([-1.02, 0.02])
    plt.show()

coord = np.array(coord,dtype=object)


import scipy.special as sc


def rhs_lhs(x,y,K,Heave=False,Diffraction=False):
    """Function to calculate the source point,
    length of the segment, normal vector into the body and the velocity potential"""
    k = len(x)
    g= 9.81
    omega = np.sqrt(g*K)
    x_bar = np.zeros(k)
    y_bar = np.zeros(k)
    dS = np.zeros(k)
    dy = np.zeros(k)
    dx = np.zeros(k)
    xm = x[:-1]
    xp = x[1:]
    ym = y[:-1]
    yp = y[1:]


    # Compute the source point and length of the segment
    for m in range(1,k):
        dy[m] = y[m]-y[m-1]
        dx[m] = x[m]-x[m-1]
        x_bar[m] = 0.5*(x[m-1]+x[m])
        y_bar[m] = 0.5*(y[m-1]+y[m])
        dS[m] = np.sqrt(dx[m]**2 + dy[m]**2)


    x_bar = x_bar[1:]
    y_bar = y_bar[1:]
    dS = dS[1:]
    dy = dy[1:]
    dx = dx[1:]
    x = x[1:]
    y = y[1:]

    # Compute the normal vector
    n1 = -dy/dS
    n2 = dx/dS
    n6 =n2*x - n1*y
    n = np.array([n1,n2,n6])

    xg1 = np.zeros(k-1)
    xg2 = np.zeros(k-1)
    yg1 = np.zeros(k-1)
    yg2 = np.zeros(k-1)

    if Diffraction == True:
        phi_0 = np.exp(K*y_bar-(K*x_bar*1j))
    else:
        phi_0 = np.exp(K*y_bar-(K*x_bar*1j))
        phi_0n = K*(n2-1j*n1)*phi_0


    # Compute two-points Gauss integration
    for m in range(k-1):
        xg1[m] = 0.5*dx[m]*np.sqrt(3)/3 + x_bar[m]
        xg2[m] = -0.5*dx[m]*np.sqrt(3)/3 + x_bar[m]
        yg1[m] = 0.5*dy[m]*np.sqrt(3)/3 + y_bar[m]
        yg2[m] = -0.5*dy[m]*np.sqrt(3)/3 + y_bar[m]


    NN = len(x_bar)
    idx = np.linspace(1,NN,NN)
    gg = np.zeros((NN,NN),dtype=np.complex_)
    ss = np.zeros((NN,NN),dtype=np.complex_)

    for i in range(NN):
        for j in range(NN):
        # rhs, log(r) term with 2pts Gauss quadrature
            xa1 = xg1[j] - x_bar[i]
            xa2 = xg2[j] - x_bar[i]
            ya1 = yg1[j] - y_bar[i]
            ya2 = yg2[j] - y_bar[i]
            ra1 = np.sqrt(xa1**2 + ya1**2)
            ra2 = np.sqrt(xa2**2 + ya2**2)
            g0 = (np.log(ra1) + np.log(ra2))*0.5

        # other term with midtpoint rule
            xa = x_bar[j] - x_bar[i]
            yb = y_bar[j] + y_bar[i]
            rb = np.sqrt(xa**2 + yb**2)
            g1 = -np.log(rb)
            zz = K*(yb - 1j*xa)
            f1 = -2*np.exp(zz)*(sc.exp1(zz) + np.log(zz) - np.log(-zz))
            f2 = 2*np.pi*np.exp(zz)
            g2 = f1.real + 1j*(f2.real)
            gg[i,j] = (g0 + g1 + g2)*dS[j]

        # lhs
            arg0 = (np.log((xm[j] - x_bar[i] + 1j*(ym[j] - y_bar[i])) /
            (xp[j] - x_bar[i] + 1j*(yp[j] - y_bar[i])))).imag
            if j==i:
                if Heave == True:
                    arg0 = -np.pi
                else:
                    arg0 = np.pi
            arg1 = (np.log((xm[j] - x_bar[i] + 1j*(ym[j] + y_bar[i])) /
            (xp[j] - x_bar[i] + 1j*(yp[j] + y_bar[i])))).imag
            help1 = (n1[j]*(f1.imag + 1j*f2.imag) + n2[j]*
            (f1.real + 1j*f2.real))*K*dS[j]
            ss[i,j] = arg0 + arg1 + help1

    if Heave == False:
        rhs = np.dot(gg,phi_0n)
        lhs = np.dot(ss,phi_0)
        return x_bar, y_bar, n, rhs, lhs, idx

    else:
        if Diffraction == True:
            rhs = -2*np.pi*phi_0
            lhs = ss
            phiD = np.linalg.solve(lhs, rhs)
            return phiD, idx, n2, dS, x_bar, y_bar,n1
        else:
            rhs = np.dot(gg,n2)
            lhs = ss
            phi_2 = np.linalg.solve(lhs, rhs)
            return phi_2, idx, n2, dS, x_bar, y_bar,n1


x1 = coord[0,0]
y1 = coord[0,1]


K = [1.2/D,0.9/D]

# Plot lhs==rhs
for i in range(len(K)):
    x_bar, y_bar, n, rhs, lhs, idx = rhs_lhs(x1,y1,K[i],Heave=False,Diffraction=False)
    plt.plot(idx,rhs.real,'y')
    plt.plot(idx,lhs.real,'r.')
    plt.plot(idx,rhs.imag,'b')
    plt.plot(idx,lhs.imag,'r+')
    plt.xlabel('N')
    plt. title('comparison between real and Imag')
    plt.legend(['RHS real','LHS real','RHS Imag','LHS Imag'])
    plt.grid(True)
    plt.show()

# Plot the velocity potential heave mode
for i in range(len(coord[:,0])):
    for j in range(len(K)):
        phi_2, idx, n2, dS, x_bar, y_bar,n1 = rhs_lhs(coord[i,0],coord[i,1],K[j],Heave=True,Diffraction=False)
        plt.plot(idx,phi_2.real)
        plt.plot(idx,phi_2.imag)
        plt.ylabel(r'$\phi_2$')
        plt.xlabel('N')
        plt. title('Velocity potential heave problem')
        plt.legend([r'$\phi_2$ real KD=%g'%K[j-1],'$\phi_2$ imag KD=%g'%K[j-1],
        '$\phi_2$ real KD=%g'%K[j],'$\phi_2$ imag KD=%g'%K[j]])
        plt.grid(True)
    plt.show()

#compute coeff
D = 1
#L = np.array([2,1,0.1])
KD = np.linspace(0.01,2,60)
f22 = np.zeros((3,len(KD)),dtype=np.complex_)
b22 = np.zeros((3,len(KD)),dtype=np.complex_)
Am = np.zeros((3,len(KD)),dtype=np.complex_)
Ap = np.zeros((3,len(KD)),dtype=np.complex_)
X2 = np.zeros((3,len(KD)),dtype=np.complex_)
X2FK = np.zeros((3,len(KD)),dtype=np.complex_)
X2H1 = np.zeros((3,len(KD)),dtype=np.complex_)
X2H2 = Am

# Compute phi_D, exciting force, damping, and added mass
for i in range(len(coord[:,0])):
    for j in range(len(KD)):
        phi_2, idx, n2, dS, x_bar, y_bar,n1 = rhs_lhs(coord[i,0],coord[i,1],KD[j],Heave=True,Diffraction=False)
        phiD, idx, n2, dS, x_bar, y_bar,n1 = rhs_lhs(coord[i,0],coord[i,1],KD[j],Heave=True,Diffraction=True)
        X2[i,j] = sum(phiD*n2*dS)
        f22[i,j] = sum(phi_2*n2*dS)
        phi0 = np.exp(KD[j]/D*(y_bar-1j*x_bar))
        Am[i,j] = 1j*sum((phi_2*(-1j*n1*KD[j]/D + KD[j]/D*n2)-n2)*phi0*dS)
        Ap[i,j] = 1j*sum((phi_2*(1j*n1*KD[j]/D + KD[j]/D*n2)-n2)*np.conjugate(phi0)*dS)
        b22[i,j] = 0.5*(Am[i,j]*np.conjugate(Am[i,j])+Ap[i,j]*np.conjugate(Ap[i,j]))
        X2FK[i,j] = L[i]*np.exp(-KD[j])*np.sin(KD[j]*L[i]/2)/(KD[j]*L[i]/2)
        X2H1[i,j] = sum((n2-phi_2*KD[j]/D*(n2-1j*n1))*phi0*dS)

    # Plot added mass and damping
    plt.plot(KD,f22[i,:].real)
    plt.plot(KD,-f22[i,:].imag,'r.',KD,-f22[i,:].imag,'k--')
    plt.xlabel(r'$\omega^2 D/g$')
    plt. title('Added mass and damping coefficient L/D = %g.' %L[i])
    plt.legend([r'$\frac{a_{22}}{\rho D^2}$',r'$\frac{b_{22}}{\rho \omega D^2}$'])
    plt.grid(True)
    plt.show()

    # Exciting force plot
    plt.plot(KD,abs(X2[i,:]))
    plt.xlabel(r'$\omega^2 D/g$')
    plt.ylabel(r'$\frac{|X_2|}{\rho g D}$')
    plt.title('Exciting force. L/D = %g.' %L[i])
    plt.grid(True)
    plt.show()


#Plot the velocity potential
for i in range(len(coord[:,0])):
    phiD, idx, n2, dS, x_bar, y_bar,n1 = rhs_lhs(coord[i,0],coord[i,1],1.2,Heave=True,Diffraction=True)
    plt.plot(idx,phiD.real,'b')
    plt.plot(idx,phiD.imag,'r')
    plt.ylabel(r'$-\frac{i \omega}{g} \phi_D$')
    plt.xlabel('N')
    plt. title('Velocity potential diffraction problem for L/D = %g'%L[i])
    plt.legend([r'real','imag'])
    plt.grid(True)
    plt.show()

# Comparison exciting force
for i in range(len(coord[:,0])):
    plt.plot(KD,abs(X2[i,:]),'b')
    plt.plot(KD,abs(X2FK[i,:]),'r')
    plt.plot(KD,abs(X2H1[i,:]),'g',alpha=0.7)
    plt.plot(KD,abs(X2H2[i,:]),'k.')
    plt.xlabel(r'$\omega^2 D/g$')
    plt.ylabel(r'$\frac{|X_2|}{\rho g}$')
    plt.title('Exciting force for L/D = %g'%L[i])
    plt.legend([r'$|X_2|$','$|X_2^{FK}|$','$|X_2^{H,1}|$','$|X_2^{H,2}|$'])
    plt.grid(True)
    plt.show()


#Response
Xi_A_P = np.zeros((3,len(KD)),dtype=np.complex_)
Xi_A_FK = np.zeros((3,len(KD)),dtype=np.complex_)
Xi_A_FK_a = np.zeros((3,len(KD)),dtype=np.complex_)

b_22FK = (abs(X2FK))**2
for i in range(len(coord[:,0])):
    for j in range(len(KD)):
        Xi_A_P[i,j] = X2[i,j]/(L[i] - KD[j]*L[i] - KD[j]*f22[i,j])
        Xi_A_FK[i,j] = X2FK[i,j]/(L[i] - KD[j]*L[i] + 1j*KD[j]*b_22FK[i,j])
        Xi_A_FK_a[i,j] = X2FK[i,j]/(L[i] - KD[j]*(L[i]+f22[i,j].real) - KD[j]*1j*b_22FK[i,j])

    plt.plot(KD,abs(Xi_A_P[i,:]),'b')
    plt.plot(KD,abs(Xi_A_FK[i,:]),'g')
    plt.plot(KD,abs(Xi_A_FK_a[i,:]),'r.')
    plt.legend([r'$|X_2|$','$|X_2^{FK}|$','$|X_2^{FK}| a_{22}$'])
    plt.xlabel(r'$\omega^2 D/g$')
    plt.ylabel(r'$\frac{|\xi_2|}{A}$',fontsize=14)
    plt.title('Response L/D = %g'%L[i])
    plt.grid(True)
    plt.show()

# Plot the comparison of b_22
for i in range(len(L)):
    plt.plot(KD,-f22[i,:].imag,alpha=0.7)
    plt.plot(KD,b22[i,:].real,'r.')
    plt.plot(KD,b_22FK[i,:].real,'g')
    plt.xlabel(r'$\omega^2 D/g$')
    plt. title('Damping coefficient L/D = %g'%L[i])
    plt.legend([r'$\frac{b_{22}}{\rho \omega D^2}$ (pressure)',
    r'$\frac{b_{22}}{\rho\omega D^2}$ (energy balance)',
    r'$\frac{b_{22}}{\rho \omega D^2}$ (FK)'])
    plt.grid(True)
    plt.show()
