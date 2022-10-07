clc
clear
close all

for ap = linspace(0, 1, 15)
    for bp = linspace(ap + 0.5, 3, 15)
        params = struct('a', ap, 'b', bp);
        fprintf('At a = %.4f, b = %.4f\n', params.a, params.b)

        % Define geometry
        crv = nrbline( [ 1 0 ] , [ 2 0 ] ) ;
        srf = nrbrevolve( crv , [ 0 0 0 ] , [ 0 0 1 ] , pi / 2 ) ;
        srf.coefs(4, :, :) = 1;
        srf.coefs(1:2, 2, 1) = params.a;
        srf.coefs(1:2, 2, 2) = params.b;
%         srf = nrbdegelev( srf , [ 1 1 ] );
%         figure
%         nrbctrlplot ( srf )
%         view(0,90)

        problem_data.geo_name = srf;
        output_file = sprintf('geo_quarter_ring_a%.4db%.4d_out_127', ...
                               round(params.a*1000), round(params.b*1000));

        % Type of boundary conditions for each side of the domain
        problem_data.nmnn_sides   = [1 2];
        problem_data.drchlt_sides = [3 4];

        % Physical parameters
        problem_data.c_diff  = @(x, y) ones(size(x));

        % Source and boundary terms
        problem_data.f = @(x, y) ones(size (x));
        problem_data.g = @(x, y, ind) zeros(size (x));
        problem_data.h = @(x, y, ind) zeros(size (x));

        % 2) CHOICE OF THE DISCRETIZATION PARAMETERS
        method_data.degree     = [4 4];       % Degree of the splines
        method_data.regularity = [2 2];       % Regularity of the splines
        method_data.nsub       = [40 40];     % Number of subdivisions
        method_data.nquad      = [4 4];       % Points for the Gaussian quadrature rule

        % 3) CALL TO THE SOLVER
        [geometry, msh, space, u] = solve_laplace_iso (problem_data, method_data);

        % 4) POST-PROCESSING
        % 4.1) EXPORT TO PARAVIEW
        vtk_pts = {linspace(0, 1, 127), linspace(0, 1, 127)};
%         fprintf ('The result is saved in the file %s \n \n', output_file);
%         sp_to_vtk (u, space, geometry, vtk_pts, output_file, 'u')

        % 4.2) PLOT IN MATLAB. COMPARISON WITH THE EXACT SOLUTION

        [eu, F] = sp_eval (u, space, geometry, vtk_pts);
%         [X, Y]  = deal (squeeze(F(1,:,:)), squeeze(F(2,:,:)));
%         contourf(X, Y, eu)

        g_nurbs = geometry.nurbs;
        save(output_file, 'F', 'vtk_pts', 'eu', 'g_nurbs', 'params')
    end
end
