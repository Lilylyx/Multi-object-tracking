function []=write_nei(csvdir, vid_name, dres)

nei = dres.nei;
py.write_nei.write_csv(csvdir, vid_name, nei)