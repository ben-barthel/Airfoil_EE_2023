function [mask_wing] = airfoil_mask(x,y,xwing,ywing)
[X,Y] = meshgrid(x,y);
[mask_wing,~] = inpolygon(X,Y,xwing,ywing);
mask_wing = 1 - mask_wing;
end