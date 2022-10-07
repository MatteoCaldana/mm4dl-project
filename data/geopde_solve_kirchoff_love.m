% EX_KL_SHELL_SCORDELIS_LO_ROOF: solve the Kirchhoff-Lovel shell model for the Scordelis Lo roof.
clc
clear
close all

% Physical parameters
params = struct('E', 4.32e8, 'nu', 0.0, 'thickness', 0.25, 'radius', 25.0, 'theta', 0.2*pi, 'length', 50);

tiledlayout(3,3, 'Padding', 'none', 'TileSpacing', 'compact'); 

for thet = linspace(0.15, 0.35, 3)
    for len = linspace(25, 45, 3)
        fprintf('At theta %.4e, length %.4e\n', thet, len)
        params.theta = thet * pi;
        params.length = len;
        output_file = sprintf('kirchoff_lovel_scrodelis_lo_t%.4dl%.4d_out_64_v2', round(thet*10000), round(len*100));

        nrb = nrbreverse (nrbcirc(params.radius, [0 0 0], pi/2 - params.theta, pi/2 + params.theta));
        srf = nrbextrude (nrbtform(nrb,vecrotx(pi/2)), [0 params.length 0]);
        problem_data.geo_name = srf;

        % figure
        % nrbctrlplot ( srf )

        % Type of boundary conditions for each side of the domain
        % Only homogeneous Dirichlet conditions have been implemented so far.
        problem_data.drchlt_sides = [3 4];
        problem_data.drchlt_components = {[1 3] [1 3]};

        problem_data.E_coeff = @(x, y, z) params.E * ones(size(x));
        problem_data.nu_coeff = @(x, y, z) params.nu * ones(size(x));
        problem_data.thickness = params.thickness;

        % Source and boundary terms
        hx = @(x, y, z) zeros(size(x));
        hy = @(x, y, z) zeros(size(x));
        hz = @(x, y, z) -90*ones(size(x));

        problem_data.f       = @(x, y, z, ind) cat(1, ...
            reshape (hx (x,y,z), [1, size(x)]), ...
            reshape (hy (x,y,z), [1, size(x)]), ...
            reshape (hz (x,y,z), [1, size(x)]));

        % Discretization parameters
        deg = 3;

        method_data.degree     = [deg deg];
        method_data.regularity = [deg-1 deg-1];
        method_data.nsub       = [10 10];
        method_data.nquad      = [deg+1 deg+1];

        % Call to solver
        tic
        [geometry, msh, space, u] = solve_kirchhoff_love_shell (problem_data, method_data);
        toc
        fprintf('Max %.4e, Mean %.4e\n\n', max(abs(u)), mean(abs(u)))

        % Postprocessing
        vtk_pts = {linspace(0, 1, 64), linspace(0, 1, 64)};
        [eu, F] = sp_eval(u, space, geometry, vtk_pts);

        % figure
        deformed_geometry = geo_deform (50*u, space, geometry);
        nexttile
        nrbkntplot(deformed_geometry.nurbs)
        title(sprintf('\\mu=(%.2f,%.2f)', params.theta, params.length))
        g_nurbs = geometry.nurbs;
        %save(output_file, 'F', 'eu', 'vtk_pts', 'g_nurbs', 'params')
    end
end


