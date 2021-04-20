function output_hash = simple_image_hash(input_image)

output_hash = sum(sum(input_image,1).^2);

end