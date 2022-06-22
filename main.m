% retrieving the 60000 digits images in a 784 (28.28) * 60000 matrix

file_pointer = fopen('train-images.idx3-ubyte', 'r');
assert(file_pointer ~= -1, ['Could not open ', 'train-images.idx3-ubyte', '']);

magic = fread(file_pointer, 1, 'int32', 0, 'ieee-be');
magic

numImages = fread(file_pointer, 1, 'int32', 0, 'ieee-be');
numImages

numRows = fread(file_pointer, 1, 'int32', 0, 'ieee-be');
numRows

numCols = fread(file_pointer, 1, 'int32', 0, 'ieee-be');
numCols

images = fread(file_pointer, inf, 'unsigned char');

fclose(file_pointer);

% attempting to set the matrix having an image for each row
% new_images = zeros(100, 784);
% for current = 1:size(new_images, 1)
%     new_images(current, 1:784) = images((current-1)*784+1:current*784);
% end
% 
% ndims(new_images)
% size(new_images)

% IMPORTANT: reshape needed because of the nature of the dataset
images = reshape(images, numCols, numRows, numImages);
images = permute(images,[2 1 3]);

size(images, 1)
size(images, 2)
size(images, 3)

images = reshape(images, size(images, 1) * size(images, 2), size(images, 3));

% normalizing the matrix for the NN
images = double(images) / 255;

colormap gray
imagesc(reshape(images(:, 1), 28, 28)); % it shows a 5



% adding the labels
file_pointer = fopen('train-labels.idx1-ubyte', 'r');
assert(file_pointer ~= -1, ['Could not open ', 'train-labels.idx1-ubyte', '']);

magic = fread(file_pointer, 1, 'int32', 0, 'ieee-be');
assert(magic == 2049, ['Bad magic number in ', 'train-labels.idx1-ubyte', '']);

numLabels = fread(file_pointer, 1, 'int32', 0, 'ieee-be');
labels = fread(file_pointer, inf, 'unsigned char');

assert(size(labels,1) == numLabels, 'Mismatch in label count');

fclose(file_pointer);

% adding the labels to the matrix
images(size(images, 1), :) = labels;
images(size(images, 1), 1) % it is 5







