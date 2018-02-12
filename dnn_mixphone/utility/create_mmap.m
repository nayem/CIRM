function [obj, m] = create_mmap(filename,dim,dType)

% column-major
obj = memmapfile(filename,'Format',{dType,[dim 1],'x'});
m = length(obj.Data);
