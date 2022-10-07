clc
clear
close all

tiledlayout(5,5, 'Padding', 'none', 'TileSpacing', 'compact'); 

av = linspace(0, 1, 5);

for i = 1:numel(av) 
    a = av(i);
    bv = linspace(a + 0.5, 3, 5);
    for j = 1:numel(bv)
         b = bv(j);
        crv = nrbline( [ 1 0 ] , [ 2 0 ] ) ;
        srf = nrbrevolve( crv , [ 0 0 0 ] , [ 0 0 1 ] , pi / 2 ) ;
        srf.coefs(4, :, :) = 1;
        srf.coefs(1:2, 2, 1) = a;
        srf.coefs(1:2, 2, 2) = b;
        %srf = nrbdegelev( srf , [ 1 1 ] );
        nexttile
        nrbctrlplot ( srf )
        view(0,90)
        title(sprintf('\\mu=(%.2f,%.2f)', a, b))
%         nrbexport ( srf , sprintf('geo_quarter_ring_a%.4db%.4d.txt', round(a*1000), round(b*1000)) ) ;
    end
end
