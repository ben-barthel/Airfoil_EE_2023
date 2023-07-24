function [x,y,t,w,xwing,ywing] = load_vorticity()
t = readNPY('../time_snap_shot.npy');
x = readNPY('../x_snap_shot.npy');
y = readNPY('../y_snap_shot.npy');
w = readNPY('../vorticity_snap_shot.npy');
xwing = readNPY('../wing_x.npy');
ywing = readNPY('../wing_y.npy');

end