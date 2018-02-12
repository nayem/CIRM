function nd = l2normalize(data)

nms=sqrt(sum(data.^2,2));
nd = data./repmat(max(nms,realmin),1,size(data,2));