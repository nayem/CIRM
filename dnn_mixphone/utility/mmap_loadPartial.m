% function data = mmap_loadPartial(obj,range,dim)
function data = mmap_loadPartial(obj,range)

% data = zeros(length(range),dim);

% for i = 1:length(range)
%     id = range(i);
%     data(i,:) = obj.Data(id).x';
% end

data = struct2array(obj.Data(range));
% data = data';
