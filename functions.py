import numpy as np
import matplotlib.pyplot as plt
import warnings

# # Solvers
def I(x, C0): # initial u(x,0)
    u = np.zeros_like(x)
    
    # where x is between 0 and 1, set to C0
    u[(x >= 0) & (x <= 1)] = C0
    return u

def forward_euler(Nt_gaps, Nx_spaces, L1, L2, C0, T=60, D=0.1, v=0.2, b0=0, bL=0, x_heart=12.5):
    # Time parameters
    Nt_points = Nt_gaps + 1
    t = np.linspace(0.,T,Nt_points)  # times at each time step
    dt = t[1] - t[0]


    # x parameters
    Nx_points = Nx_spaces + 1 
    x = np.linspace(L1, L2, Nx_points)    # mesh points in space
    dx = x[1] - x[0]

    # Check our conditions
    C = D*dt/(dx**2)
    A = v*dt/(2*dx)

    # print(f"Delta x = {round(dx, 4)}\nDelta t = {round(dt, 4)}\nC = {round(C, 4)}\nA = {round(A, 4)}")

    if C >= 0.5:
        warnings.warn(f'C is greater than 0.5, C = {round(C,4)}')

    if A > 1:
        warnings.warn(f'A is greater than 1, A = {round(A,4)}')

    # Boundary conditions
    # dirichlet
    # b0 = 0
    # bL = 0
    
    # set up matrices for single time solutions and the full solution
    u_old = np.zeros(Nx_points)

    U = np.zeros((Nx_points,Nt_points))

    # Data structures for the linear system
    A_mat = np.zeros((Nx_points, Nx_points))

    for i in range(1, Nx_points-1):
        A_mat[i,i-1] = C + A
        A_mat[i,i+1] = C - A
        A_mat[i,i] = 1 - 2*C

    # implement the (constant-in-time) Dirichlet conditions
    A_mat[0,0] = 1
    A_mat[-1,-1] = 1

    # Set initial condition u(x,0) = I(x)
    u_old = I(x, C0)

    # initialise matrices for storing solutions
    U[:,0] = u_old

    # do timestepping 
    for n in range(1, Nt_points):
        
        # update u by solving the matrix system Au = b
        u_new = np.matmul(A_mat,u_old)
        
        # Update u_old before next step
        u_old = u_new
        U[:,n] = u_new
    
    # Find the final concentration at the heart
    index_closest = (np.abs(x - x_heart)).argmin()
    Cf = U[index_closest, -1]
    
    return Cf, U, x, t, C, A

def backward_euler(Nt_gaps, Nx_spaces, L1, L2, C0, T=60, D=0.1, v=0.2, b0=0, bL=0, x_heart=12.5):
# Time parameters
    Nt_points = Nt_gaps + 1
    t = np.linspace(0.,T,Nt_points)  # times at each time step
    dt = t[1] - t[0]


    # x parameters
    Nx_points = Nx_spaces + 1 
    x = np.linspace(L1, L2, Nx_points)    # mesh points in space
    dx = x[1] - x[0]

    # Check our conditions
    C = D*dt/(dx**2)
    A = v*dt/(2*dx)

    # print(f"Delta x = {round(dx, 4)}\nDelta t = {round(dt, 4)}\nC = {round(C, 4)}\nA = {round(A, 4)}")

    # if C >= 0.5:
    #     warnings.warn(f'C is greater than 0.5, C = {round(C,4)}')

    # if A > 1:
    #     warnings.warn(f'A is greater than 1, A = {round(A,4)}')

    # Boundary conditions
    # dirichlet
    # b0 = 0
    # bL = 0
    
    # set up matrices for single time solutions and the full solution
    u   = np.zeros(Nx_points)
    u_old = np.zeros(Nx_points)

    U = np.zeros((Nx_points,Nt_points))
    U_ex = np.zeros((Nx_points, Nt_points))

    # Data structures for the linear system
    A_mat = np.zeros((Nx_points, Nx_points))
    b = np.zeros(Nx_points)

    for i in range(1, Nx_points-1):
        A_mat[i,i-1] = -C - A
        A_mat[i,i+1] = -C + A
        A_mat[i,i] = 1 + 2*C
        
    # implement the (constant-in-time) Dirichlet conditions
    A_mat[0,0] = 1
    A_mat[Nx_points-1,Nx_points-1] = 1

    # do not use the inverse method below - it is numerically unstable but is included here
    # for comparison purposes.
    # Find the inverse of A which doesn't change in time
    # Ainv = np.linalg.inv(A)

    # Set initial condition u(x,0) = I(x)
    u_old = I(x, C0)

    # initialise matrices for storing solutions
    U[:,0] = u_old

    # do timestepping 
    for n in range(1, Nt_points):
        
        # Compute b and solve linear system
        b[1:Nx_points-2] = u_old[1: Nx_points-2]  # internal values
        b[0] = b0  # boundary conditions  
        b[-1] = bL
        
        # update u by calculating A_inverse . b - again, don't do this!
        #u_new = np.dot(Ainv,b)

        # update u by solving the matrix system Au = b
        u_new = np.linalg.solve(A_mat,b)
        
        # Update u_old before next step
        u_old = u_new
        U[:,n] = u_new    
    
    # Find the final concentration at the heart
    index_closest = (np.abs(x - x_heart)).argmin()
    Cf = U[index_closest, -1]
    
    return Cf, U, x, t


# # Plotting
def concentration_x_plot(x, U, C0, heart_loc, t_str, ax, xlim_right=15, ylims = None):
    # Create a figure outside of the call to function
    
    # heart_loc = 12.5
    index_closest = (np.abs(x - heart_loc)).argmin()
    C_heart = U[index_closest]
    # print(index_closest)

    # Clear axes for next iteration of animation loop
    # ax.clear() 
    
    ax.plot(x, U, '.')
    
    ax.set_xlabel('x [m]')
    ax.set_ylabel('Concentration [g/m]')
    
    # title_str = f'{t_str}, $C_0$ = {C0} g/m'
    # ax.set_title(title_str)

    ax.set_xlim([0,xlim_right])
    
    if ylims:
        ax.set_ylim(list(ylims))
    
    ax.axvline(x=heart_loc, color='r', linestyle='--')
    
    ax.axhline(y=C_heart, color='b', linestyle='--', label= f'Cf = {round(C_heart, 4)}')
    
    ax.text(x=heart_loc+0.15, y=C_heart, s=f'Cf = {round(C_heart, 4)}', color='b', va='bottom', ha='left')
    # ax.legend()
    