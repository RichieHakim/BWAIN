function image = maskImage(image, border_outer, border_inner)
    border_outer = int64(border_outer);
    border_inner = int64(border_inner);
    imDim = size(image);
    mid = floor(imDim./2);

    image(1:border_outer,:) = 0;
    image(end-border_outer:end,:) = 0;
    image(:, 1:border_outer) = 0;
    image(:, end-border_outer:end) = 0;

    image(mid(1)-border_inner:mid(1)+border_inner, mid(2)-border_inner:mid(2)+border_inner) = 0;
end