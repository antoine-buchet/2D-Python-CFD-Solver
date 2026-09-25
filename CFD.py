import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numba import njit

print("Choose geometry:")
print("  1  NACA 0012 airfoil")
print("  2  Cylinder")
print("  3  Flat plate")
print("  4  Rectangle")

while True:
    gchoice = input("\nSelect geometry [1/2/3/4, default=1]: ").strip() or "1"
    if gchoice in ("1", "2", "3", "4"):
        break
    print("Please enter 1, 2, 3, or 4.")

geometry_names = {"1": "NACA 0012", "2": "Cylinder", "3": "Flat plate", "4": "Rectangle"}
print(f"\n{geometry_names[gchoice]} selected.\n")

rho = 1.204 #density of air at 1 atm, 20°C (kg/m^3)
nu = 1.5e-3 #kinematic viscosity (m^2/s)
L = 4 #domain length (m)
H = 1 #domain height (m)
U = 1.0 #inlet airflow speed (m/s)
alpha = 5.0 #airfoil angle of attack (°)
chord = 0.6 #airfoil chord length (m)
dx = dy = 0.01
dt = 0.002
Nt = 4000

while True:
    tend = Nt * dt
    CFL  = U * dt / dx
    cfl_note = "stable" if CFL <= 1 else "WARNING: CFL > 1, may be unstable"

    print("\n" + "─" * 60)
    print("Simulation parameters")
    print("─" * 60)
    print(f"  Geometry       : {geometry_names[gchoice]}")
    print(f"  Inlet speed  U : {U} m/s")
    print(f"  Angle of attack: {alpha} deg")
    print(f"  Domain         : {L} m x {H} m")
    print(f"  Grid spacing   : dx = dy = {dx} m")
    print(f"  Time step   dt : {dt} s")
    print(f"  Timesteps   Nt : {Nt}")
    print(f"  End time  tend : {tend:.3f} s")
    print(f"  CFL number     : {CFL:.3f}  ({cfl_note})")
    print("─" * 60)
    print("\nOptions:")
    print("  y     -- confirm and run")
    print("  u     -- change inlet speed U")
    print("  alpha -- change angle of attack")
    print("  dx    -- change grid spacing (dt auto-adjusted to keep CFL=0.2)")
    print("  dt    -- change time step")
    print("  nt    -- change number of timesteps")

    confirm = input("\n  Your choice [y/u/alpha/dx/dt/nt]: ").strip().lower()

    if confirm == "y" or confirm == "":
        print()
        break

    elif confirm == "u":
        while True:
            try:
                new_U = float(input(f"Enter new U in m/s (current: {U}): ").strip())
                if new_U <= 0:
                    raise ValueError
                U = new_U
                # auto-adjust dt so CFL stays at 0.2
                dt = round(0.2 * dx / U, 6)
                print(f"dt auto-set to {dt} s to keep CFL = 0.2")
                break
            except ValueError:
                print("Invalid -- please enter a positive number.")

    elif confirm == "alpha":
        while True:
            try:
                new_alpha = float(input(f"Enter new angle of attack in deg (current: {alpha}): ").strip())
                if not (-90 < new_alpha < 90):
                    raise ValueError
                alpha = new_alpha
                break
            except ValueError:
                print("Invalid -- please enter a value between -90 and 90 deg.")

    elif confirm == "dx":
        while True:
            try:
                new_dx = float(input(f"  Enter new dx in m (current: {dx}): ").strip())
                if new_dx <= 0:
                    raise ValueError
                dx = new_dx
                dy = dx
                # auto-adjust dt so CFL stays at 0.2
                dt = round(0.2 * dx / U, 6)
                print(f"dt auto-set to {dt} s to keep CFL = 0.2")
                break
            except ValueError:
                print("Invalid -- please enter a positive number.")

    elif confirm == "dt":
        while True:
            try:
                new_dt = float(input(f"Enter new dt (current: {dt}): ").strip())
                if new_dt <= 0:
                    raise ValueError
                dt = new_dt
                break
            except ValueError:
                print("Invalid -- please enter a positive number.")

    elif confirm == "nt":
        while True:
            try:
                new_nt = int(input(f"Enter new Nt (current: {Nt}): ").strip())
                if new_nt <= 0:
                    raise ValueError
                Nt = new_nt
                break
            except ValueError:
                print("Invalid -- please enter a positive integer.")

    else:
        print("Please enter y, u, alpha, dx, dt, or nt.")

dy = dx
theta = np.deg2rad(-alpha)
tend = Nt * dt

x = np.arange(0, L + dx, dx)
y = np.arange(0, H + dy, dy)
Nx = len(x)
Ny = len(y)

X, Y = np.meshgrid(x, y, indexing='ij')

print(f"  Grid : {Nx} x {Ny} = {Nx*Ny:,} nodes")
print(f"  dt   : {dt}  |  Nt : {Nt}  |  tend : {tend:.2f} s")
CFL = U * dt / dx
print(f"  CFL  : {CFL:.3f}  ({'stable' if CFL <= 1 else 'WARNING > 1'})\n")

def build_geometry(gchoice, X, Y, x, y, L, H, theta, chord, dx, dy):
    Nx, Ny = X.shape

    #Airfoil
    if gchoice == "1":
        c = 0.8 #chord length (m)
        x_le = 0.7 #leading edge x position
        y_mid = H / 2
        t_c = 0.12 #thickness-to-chord ratio

        xa   = np.linspace(0, 1, 400)
        yt1d = 5*t_c*(0.2969*np.sqrt(xa) - 0.1260*xa
                      - 0.3516*xa**2 + 0.2843*xa**3 - 0.1015*xa**4)

        center  = 0.5 * c
        x_local = c * np.concatenate([xa, xa[::-1]])
        y_local = c * np.concatenate([yt1d, -yt1d[::-1]])

        xr_loc  = (x_local - center)*np.cos(theta) - y_local*np.sin(theta)
        yr_loc  = (x_local - center)*np.sin(theta) + y_local*np.cos(theta)

        x_coords = x_le + center + xr_loc
        y_coords = y_mid + yr_loc

        #nudge inside domain if needed
        margin  = max(dx, dy) * 3
        shift_x = shift_y = 0.0
        if np.min(x_coords) < x.min() + margin:
            shift_x = x.min() + margin - float(np.min(x_coords))
        elif np.max(x_coords) > x.max() - margin:
            shift_x = x.max() - margin - float(np.max(x_coords))
        if np.min(y_coords) < y.min() + margin:
            shift_y = y.min() + margin - float(np.min(y_coords))
        elif np.max(y_coords) > y.max() - margin:
            shift_y = y.max() - margin - float(np.max(y_coords))
        x_coords += shift_x;  y_coords += shift_y
        x_le     += shift_x;  y_mid    += shift_y

        Xc = X - (x_le + center);  Yc = Y - y_mid
        Xr =  Xc*np.cos(theta) + Yc*np.sin(theta)
        Yr = -Xc*np.sin(theta) + Yc*np.cos(theta)

        x_rel = (Xr + center) / c
        y_rel = Yr / c

        yt   = np.zeros_like(X)
        mk   = (x_rel >= 0) & (x_rel <= 1)
        yt[mk] = 5*t_c*(0.2969*np.sqrt(x_rel[mk]) - 0.1260*x_rel[mk]
                        - 0.3516*x_rel[mk]**2 + 0.2843*x_rel[mk]**3
                        - 0.1015*x_rel[mk]**4)
        mask = mk & (y_rel >= -yt) & (y_rel <= yt)
        return mask, x_coords, y_coords, "NACA 0012"

    #Cylinder
    elif gchoice == "2":
        x0 = L / 4 #centre x — placed at quarter-domain
        y0 = H / 2 #centre y — mid-height
        R  = 0.1 #radius (m)
        mask = (X - x0)**2 + (Y - y0)**2 < R**2
        phi  = np.linspace(0, 2*np.pi, 300)
        x_coords = x0 + R*np.cos(phi)
        y_coords = y0 + R*np.sin(phi)
        return mask, x_coords, y_coords, "Cylinder"

    #Flat plate
    elif gchoice == "3":
        x0 = 0.8 #leading edge x
        y0 = H / 2 #leading edge y (mid-height)
        thickness = max(0.03, 2.0 * dx) #half-thickness [m]
        dist = (np.abs((Y - y0) - (X - x0)*np.tan(theta)) / np.sqrt(1 + np.tan(theta)**2)) #perpendicular distance from the plate centreline
        proj = (X - x0)*np.cos(theta) + (Y - y0)*np.sin(theta) #projection along the plate axis
        mask = (dist <= thickness) & (proj >= 0) & (proj <= chord)
        #outline: rectangle in rotated frame
        corners_s = np.array([0, chord, chord, 0, 0])   # along plate
        corners_n = np.array([-thickness, -thickness, thickness,  thickness, -thickness])
        x_coords = x0 + corners_s*np.cos(theta) - corners_n*np.sin(theta)
        y_coords = y0 + corners_s*np.sin(theta) + corners_n*np.cos(theta)
        return mask, x_coords, y_coords, "Flat plate"

    #Rectangle
    else:
        w = 0.3 #width (m)
        h = 0.2 #height (m)
        x1 = L/4 - w/2
        x2 = L/4 + w/2
        y1 = H/2 - h/2
        y2 = H/2 + h/2
        mask = (X >= x1) & (X <= x2) & (Y >= y1) & (Y <= y2)
        x_coords = np.array([x1, x2, x2, x1, x1])
        y_coords = np.array([y1, y1, y2, y2, y1])
        return mask, x_coords, y_coords, "Rectangle"


object_mask, x_coords, y_coords, geom_label = build_geometry(gchoice, X, Y, x, y, L, H, theta, chord, dx, dy)

def velocity_boundary_conditions(arr, U0, component):
    arr[0,  :] = U0 if component == 'u' else 0 #Dirichlet condition at inlet
    arr[-1, :] = arr[-2, :] #Neumann condition at outlet
    arr[:,  0] = arr[:,  1] #Neumann condition at bottom wall
    arr[:, -1] = arr[:, -2] #Neumann condition at top wall
    arr[object_mask] = 0 #Dirichlet condition at object boundary
    return arr

u = np.zeros((Nx, Ny))
v = np.zeros((Nx, Ny))
p = np.zeros((Nx, Ny))
u[object_mask] = 0
v[object_mask] = 0
u = velocity_boundary_conditions(u, U, 'u')
v = velocity_boundary_conditions(v, U, 'v')

v[int(Nx/4):int(Nx/3), int(Ny/2):int(Ny/2)+3] += 0.02 * U #small perturbation to break symmetry and trigger vortex shedding

error_criterion = 1e-4
omega = 1.5 #SOR Gauss-Seidel relaxation factor

@njit(cache=True)
def solve_pressure_gauss_seidel(p, dundx, dvndy, error_criterion, object_mask, dx, dy, dt, rho, Nx, Ny, omega):
    dx2   = dx * dx
    dy2   = dy * dy
    denom = 2.0 * (dx2 + dy2)
    for _ in range(10000):
        max_change = 0.0

        for i in range(1, Nx - 1):
            for j in range(1, Ny - 1):
                if not object_mask[i, j]:
                    p_gs = (dy2 * (p[i+1, j] + p[i-1, j])
                            + dx2 * (p[i, j+1] + p[i, j-1])
                            - dx2 * dy2 * (rho / dt)
                            * (dundx[i, j] + dvndy[i, j])) / denom

                    p_new  = (1.0 - omega) * p[i, j] + omega * p_gs
                    change = abs(p_new - p[i, j])
                    if change > max_change:
                        max_change = change
                    p[i, j] = p_new

        for j in range(Ny):
            p[0,  j] = p[1,  j] #Neumann condition at inlet
            p[-1, j] = 0 #Dirichlet condition at oulet
        for i in range(Nx):
            p[i,  0] = p[i,  1] #Neumann condition at bottom wall
            p[i, -1] = p[i, -2] #Neumann condition at top wall

        #Neumann condition on object boundary
        for i in range(1, Nx - 1):
            for j in range(1, Ny - 1):
                if object_mask[i, j]:
                    nb_sum = 0.0;  nb_n = 0
                    if not object_mask[i-1, j]: nb_sum += p[i-1, j]; nb_n += 1
                    if not object_mask[i+1, j]: nb_sum += p[i+1, j]; nb_n += 1
                    if not object_mask[i, j-1]: nb_sum += p[i, j-1]; nb_n += 1
                    if not object_mask[i, j+1]: nb_sum += p[i, j+1]; nb_n += 1
                    if nb_n > 0:
                        p[i, j] = nb_sum / nb_n

        if max_change < error_criterion:
            break

    return p, max_change

@njit(cache=True)
def velocity_star(u, v, nu, dt, dx, dy, object_mask, Nx, Ny):
    un = u.copy()
    vn = v.copy()
    for i in range(1, Nx - 1):
        for j in range(1, Ny - 1):
            dudx = ((u[i, j] - u[i-1,j])/dx if u[i, j] > 0 else (u[i+1,j] - u[i, j])/dx)
            dudy = ((u[i, j] - u[i,j-1])/dy if v[i, j] > 0 else (u[i,j+1] - u[i, j])/dy)
            lap_u = ((u[i+1,j] - 2*u[i, j] + u[i-1,j])/(dx*dx) + (u[i,j+1] - 2*u[i, j] + u[i,j-1])/(dy*dy))
            un[i,j] = u[i, j] + dt*(-u[i, j]*dudx - v[i, j]*dudy + nu*lap_u)

            dvdx = ((v[i, j] - v[i-1,j])/dx if u[i, j] > 0 else (v[i+1,j] - v[i, j])/dx)
            dvdy = ((v[i, j] - v[i,j-1])/dy if v[i, j] > 0 else (v[i,j+1] - v[i, j])/dy)
            lap_v = ((v[i+1,j] - 2*v[i, j] + v[i-1,j])/(dx*dx) + (v[i,j+1] - 2*v[i, j] + v[i,j-1])/(dy*dy))
            vn[i,j] = v[i, j] + dt*(-u[i, j]*dvdx - v[i, j]*dvdy + nu*lap_v)
    return un, vn, dudx, dudy, dvdx, dvdy

@njit(cache=True)
def compute_divergence(un, vn, dx, dy, Nx, Ny):
    dundx = np.zeros((Nx, Ny))
    dvndy = np.zeros((Nx, Ny))
    for i in range(1, Nx - 1):
        for j in range(1, Ny - 1):
            dundx[i,j] = (un[i+1,j] - un[i-1,j]) / (2*dx)
            dvndy[i,j] = (vn[i,j+1] - vn[i,j-1]) / (2*dy)
    return dundx, dvndy

@njit(cache=True)
def velocity_correction(u, v, un, vn, p, dt, rho, dx, dy, object_mask, Nx, Ny):
    for i in range(1, Nx - 1):
        for j in range(1, Ny - 1):
            if not object_mask[i, j]:
                dpdx = (p[i+1, j] - p[i-1, j]) / (2.0 * dx)
                dpdy = (p[i, j+1] - p[i, j-1]) / (2.0 * dy)
                u[i, j] = un[i, j] - (dt / rho) * dpdx
                v[i, j] = vn[i, j] - (dt / rho) * dpdy
    return u, v

def CFD(u, v, p):
    converged = True
    snapshots = [] #(t, u, v, p) tuples
    n_snapshots = 60 #frames to store
    x_geom_min = float(np.min(x[object_mask.any(axis=1)]))
    t_arrival = 0.8 * x_geom_min / max(U, 1e-6) #80% of travel time
    step_start = max(1, int(t_arrival / dt)) #first step to save
    steps_left = max(1, Nt - step_start) #steps remaining
    save_every = max(1, steps_left // n_snapshots) #interval between saves
    print(f"  Flow arrival at geometry: t = {t_arrival:.2f} s "
          f"(step {step_start})  |  saving every {save_every} steps\n")

    for step in range(1, Nt + 1):
        un, vn, dudx, dudy, dvdx, dvdy = velocity_star(u, v, nu, dt, dx, dy, object_mask, Nx, Ny)
        un = velocity_boundary_conditions(un, U, 'u')
        vn = velocity_boundary_conditions(vn, U, 'v')

        dundx, dvndy = compute_divergence(un, vn, dx, dy, Nx, Ny)

        p, residual = solve_pressure_gauss_seidel(p, dundx, dvndy, error_criterion, object_mask, dx, dy, dt, rho, Nx, Ny, omega)

        if residual > 1e6 or np.isnan(residual):
            print(f"\n  ERROR: Pressure solver diverged at step {step}.")
            print(f"  Residual = {residual:.3e}. Try reducing dt or U.\n")
            converged = False
            break

        if np.isnan(u).any() or np.isinf(u).any():
            print(f"\n  ERROR: NaN/Inf in velocity at step {step}.\n")
            converged = False
            break

        velocity_correction(u, v, un, vn, p, dt, rho, dx, dy, object_mask, Nx, Ny)
        u = velocity_boundary_conditions(u, U, 'u')
        v = velocity_boundary_conditions(v, U, 'v')

        if step >= step_start and step % save_every == 0:
            snapshots.append((step * dt, u.copy(), v.copy(), p.copy()))

        pct = (step / Nt) * 100
        if step == 1 or (step % max(1, Nt//20) == 0):
            maxu = float(np.nanmax(np.abs(u)))
            maxv = float(np.nanmax(np.abs(v)))
            print(f"  {pct:5.1f}%  t={step*dt:.3f}s  U={U:.2f}  max|u|={maxu:.3e}  max|v|={maxv:.3e}  CFL={U*dt/dx:.3f}  p_res={residual:.2e}")

    status = "Complete" if converged else "Aborted (non-convergence)"
    print(f"\n  {status} {len(snapshots)} snapshots\n")
    return u, v, p, snapshots, dudx, dudy, dvdx, dvdy

u, v, p, snapshots, dudx, dudy, dvdx, dvdy = CFD(u, v, p)

if snapshots:
    p_final = snapshots[-1][3]
    p_abs   = float(np.nanpercentile(np.abs(p_final), 99))

    if p_abs < 1e-8:
        p_all  = np.concatenate([s[3].ravel() for s in snapshots])
        p_abs  = float(np.nanpercentile(np.abs(p_all), 98))
    if p_abs < 1e-8:
        p_abs = 1.0

    vmin = -p_abs #symmetric: blue = suction / low pressure
    vmax =  p_abs #symmetric: red  = stagnation / high pressure

    fig_ani, ax_ani = plt.subplots(figsize=(11, 3.5))
    fig_ani.tight_layout(pad=1.5)

    #Colourbar is created once and reused; we update its mappable each frame.
    _mesh_init = ax_ani.pcolormesh(x, y, snapshots[0][3].T, cmap='plasma', vmin=vmin, vmax=vmax, shading='auto')
    cbar = plt.colorbar(_mesh_init, ax=ax_ani, label='Pressure (Pa)')

    def draw_frame(frame):
        t_val, u_s, v_s, p_s = snapshots[frame]
        ax_ani.cla() #clear axes to remove previous streamlines
        pcm = ax_ani.pcolormesh(x, y, p_s.T, cmap='plasma', vmin=vmin, vmax=vmax, shading='auto')

        cbar.update_normal(pcm) #Update the colourbar mappable to track this frame's mesh

        ax_ani.streamplot(x, y, u_s.T, v_s.T, color='k', density=2.5, linewidth=0.6, arrowsize=0.5)

        ax_ani.fill(x_coords, y_coords, color='dimgrey', edgecolor='black', linewidth=1.5, zorder=10) #Geometry outline

        ax_ani.set_xlim(x[0], x[-1])
        ax_ani.set_ylim(y[0], y[-1])
        ax_ani.set_xlabel('x (m)')
        ax_ani.set_ylabel('y (m)')
        ax_ani.set_aspect('equal')
        ax_ani.set_title(f"{geom_label}, U = {U} m/s, α = {alpha}°, t = {t_val:.3f} s")

    plt.tight_layout()
    ani = animation.FuncAnimation(fig_ani, draw_frame, frames=len(snapshots), interval=150, blit=False, repeat=True)
    

    save_gif = input("\nSave animation as GIF? [y/N]: ").strip().lower()
    if save_gif == 'y':
        import os, datetime

        def _slug(s):
            return str(s).lower().replace(' ', '_').replace('-', '_').replace('.', 'p')

        parameter_str = (f"geom-{_slug(geom_label)}"
            f"_U-{_slug(U)}ms"
            f"_alpha-{_slug(alpha)}deg"
            f"_dx-{_slug(dx)}m"
            f"_dt-{_slug(dt)}s"
            f"_Nt-{Nt}"
            f"_tend-{_slug(round(tend, 3))}s")

        # Append a timestamp so repeated runs never overwrite each other
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"cfd_{parameter_str}_{timestamp}.gif"

        print(f"  Saving {filename} ...", flush=True)
        ani.save(filename, writer='pillow', fps=8)
        print(f"  Saved: {filename}")
        print("A gif file has been saved to the folder were this code is located, the gif is an animation of how the flows moves from t0 to tend")
    plt.show()

fig2, ax2 = plt.subplots(figsize=(11, 3.5))
ax2.streamplot(x, y, u.T, v.T, color='k', density=4, linewidth=0.6, arrowsize=0.5)
p_abs_final = float(np.nanpercentile(np.abs(p), 99)) or 1.0
ax2.pcolormesh(x, y, p.T, cmap='plasma', vmin=-p_abs_final, vmax=p_abs_final, shading='auto')
plt.colorbar(ax2.collections[0], ax=ax2, label='Pressure (Pa)')
ax2.fill(x_coords, y_coords, color='white', edgecolor='black', linewidth=2, zorder=10)
ax2.set_title(f"{geom_label}, U = {U} m/s, t = {tend} s, α = {alpha}°")
ax2.set_xlabel('x (m)')
ax2.set_ylabel('y (m)')
ax2.set_aspect('equal')
plt.tight_layout()
plt.show()