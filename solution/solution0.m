%% Initialization

% load the image
raw = imread('../data/banana_slug.tiff');

% print its size: 2856 x 4290
[Ysize, Xsize] = size(raw);

% print its data type: uint16, therefore 16 bits
%%
class(raw);

% convert to double
raw = double(raw);

%% Linearization

% apply linear transform
black = 2047;
saturation = 15000;
lin_bayer = (raw - black) / (saturation - black);

% clip values outside of range
lin_bayer = max(0, min(lin_bayer, 1));

% Print the raw image post linearization
im_intermediate = lin_bayer * 5;
figure; imshow(im_intermediate); title("Initial Raw Image");

%% Identifying the correct Bayer pattern

% form quarter-resolution sub-images
%% 
im1 = lin_bayer(1:2:end, 1:2:end);
im2 = lin_bayer(1:2:end, 2:2:end);
im3 = lin_bayer(2:2:end, 1:2:end);
im4 = lin_bayer(2:2:end, 2:2:end);

% form quarter-resolution RGB images for each Bayer pattern type
%% 
im_grbg = cat(3, im2, im1, im3);
im_rggb = cat(3, im1, im2, im4);
im_bggr = cat(3, im4, im2, im1);
im_gbrg = cat(3, im3, im1, im2);

% Display the RGB images of each Bayer pattern type
figure(2); subplot(2,2,1); imshow(im_grbg * 4); title('grbg');
figure(2); subplot(2,2,2); imshow(im_rggb * 4); title('rggb');
figure(2); subplot(2,2,3); imshow(im_bggr * 4); title('bggr');
figure(2); subplot(2,2,4); imshow(im_gbrg * 4); title('gbrg');

% rggb looks more natural, so we're keeping it.

%% White balancing

% first, get the pixels of each color channel of the 'rggb' pattern
red = lin_bayer(1:2:end, 1:2:end);
green1 = lin_bayer(1:2:end, 2:2:end);
green2 = lin_bayer(2:2:end, 1:2:end);
blue = lin_bayer(2:2:end, 2:2:end);

% white balancing under gray world assumption:
%	compute the means of each channel, note that for green we need to use
%	both green subimages
red_mean = mean(red(:));
green_mean = mean([green1(:); green2(:)]);
blue_mean = mean(blue(:));

%	create new image and assign white-balanced values
im_gw = zeros(size(lin_bayer));
im_gw(1:2:end, 1:2:end) = red * green_mean / red_mean;
im_gw(1:2:end, 2:2:end) = green1;
im_gw(2:2:end, 1:2:end) = green2;
im_gw(2:2:end, 2:2:end) = blue * green_mean / blue_mean;

% white balancing under white world assumption:
%	compute the means of each channel, note that for green we need to use
%	both green subimages
red_max = max(red(:));
green_max = max([green1(:); green2(:)]);
blue_max = max(blue(:));

%	create new image and assign white-balanced values
im_ww = zeros(size(lin_bayer));
im_ww(1:2:end, 1:2:end) = red * green_max / red_max;
im_ww(1:2:end, 2:2:end) = green1;
im_ww(2:2:end, 1:2:end) = green2;
im_ww(2:2:end, 2:2:end) = blue * green_max / blue_max;

%% Demosaicing

% select im to use for this part
% Image white balancing under white world assumption
%im = im_ww;

% Image white balancing under gray world assumption
im = im_gw; 

% demosaic the red channel
[Y, X] = meshgrid(1:2:Xsize, 1:2:Ysize);
vals = im(1:2:end, 1:2:end);

dms = zeros(size(im));
dms(1:2:end, 1:2:end) = vals;

[Yin, Xin] = meshgrid(2:2:Xsize, 1:2:Ysize);
dms(2:2:end, 1:2:end) = interp2(Y, X, vals, Yin, Xin);
[Yin, Xin] = meshgrid(1:2:Xsize, 2:2:Ysize);
dms(1:2:end, 2:2:end) = interp2(Y, X, vals, Yin, Xin);
[Yin, Xin] = meshgrid(2:2:Xsize, 2:2:Ysize);
dms(2:2:end, 2:2:end) = interp2(Y, X, vals, Yin, Xin);

red_dms = dms;

% demosaic the blue channel
[Y, X] = meshgrid(2:2:Xsize, 2:2:Ysize);
vals = im(2:2:end, 2:2:end);

dms = zeros(size(im));
dms(2:2:end, 2:2:end) = vals;    

[Yin, Xin] = meshgrid(1:2:Xsize, 1:2:Ysize);
dms(1:2:end, 1:2:end) = interp2(Y, X, vals, Yin, Xin);
[Yin, Xin] = meshgrid(1:2:Xsize, 2:2:Ysize);
dms(1:2:end, 2:2:end) = interp2(Y, X, vals, Yin, Xin);
[Yin, Xin] = meshgrid(2:2:Xsize, 1:2:Ysize);
dms(2:2:end, 1:2:end) = interp2(Y, X, vals, Yin, Xin);

blue_dms = dms;

% demosaic the green channel
[Y1, X1] = meshgrid(1:2:Xsize, 2:2:Ysize);
vals1 = im(1:2:end, 2:2:end);

[Y2, X2] = meshgrid(2:2:Xsize, 1:2:Ysize);
vals2 = im(2:2:end, 1:2:end);

dms = zeros(size(im));
dms(1:2:end, 2:2:end) = vals1;
dms(2:2:end, 1:2:end) = vals2;


[Yin, Xin] = meshgrid(1:2:Xsize, 1:2:Ysize);
dms(1:2:end, 1:2:end) = (interp2(Y1, X1, vals1, Yin, Xin)... 
						+ interp2(Y2, X2, vals2, Yin, Xin)) / 2;
[Yin, Xin] = meshgrid(2:2:Xsize, 2:2:Ysize);
dms(2:2:end, 2:2:end) = (interp2(Y1, X1, vals1, Yin, Xin)...
						+ interp2(Y2, X2, vals2, Yin, Xin)) / 2;

green_dms = dms;

im_rgb = cat(3, red_dms, green_dms, blue_dms);

%% Brightness adjustment and gamma correction

im_gray = rgb2gray(im_rgb);
percentage = 5;
im_rgb_brightened = im_rgb * percentage * max(im_gray(:));

im_final = zeros(size(im_rgb_brightened));
inds = (im_rgb_brightened <= 0.0031308);
im_final(inds) = 12.92 * im_rgb_brightened(inds);
im_final(~inds) = real(1.055 * im_rgb_brightened(~inds) .^ (1 / 2.4) - 0.055);

% Display the Processed original image 
figure; imshow(im_final); title("Processed Original Image");

%% Writing the Image to PNG and JPEG files

% Write the Processed Original Image to PNG files
imwrite(im_final,'../data/Output_PNG.png','png');

% Write Processed Original Image to JPEG with NO compression 
imwrite(im_final,'../data/Output_JPEG.jpeg','jpeg');

% Write Processed Original Image to JPEG with Compression:Quality factor-95
imwrite(im_final,'../data/compressed_Output_JPEG_95.jpeg','jpeg','Quality', 95);

% Write the JPEG Image in compressed files to Analyse the Quality factor #
%imwrite(im_final,'../data/compressed_Output_JPEG_20.jpeg','jpeg','Quality', 20);
%imwrite(im_final,'../data/compressed_Output_JPEG_30.jpeg','jpeg','Quality', 30);
%imwrite(im_final,'../data/compressed_Output_JPEG_40.jpeg','jpeg','Quality', 40);
%imwrite(im_final,'../data/compressed_Output_JPEG_50.jpeg','jpeg','Quality', 50);
%imwrite(im_final,'../data/compressed_Output_JPEG_60.jpeg','jpeg','Quality', 60);
%imwrite(im_final,'../data/compressed_Output_JPEG_70.jpeg','jpeg','Quality', 70);
%imwrite(im_final,'../data/compressed_Output_JPEG_80.jpeg','jpeg','Quality', 80);

% Using code to find the least quality factor for compression 
ssimVal = zeros(1,9);
qualityStep = 10:10:90; % step interval of 10
figure(4); title("Finding optimum compression quality factor of image - Wait time: 2 mins.");
figure(4); plot(qualityStep,ssimVal); %empty
for i = 1:9
    imwrite(im_final,'../data/quality_compressed_Output_JPEG.jpeg','jpeg','Quality', qualityStep(i));
    im = imread('../data/quality_compressed_Output_JPEG.jpeg', 'jpeg');
    ssimVal(i) = ssim(im, uint8(im_final));
end

% PLOT to Observe the least Quality factor, for which compressed image is 
% similar to the Processed-Original-image
figure(4); plot(qualityStep,ssimVal,'b-o');
xlabel('Compression Quality');
ylabel('SSIM Value');
delete('../data/quality_compressed_Output_JPEG.jpeg'); % cleanup

% Optional: To display this compressed Image with the least Quality factor.
% On observing the plot, Quality factor 30 looks to be the lowest setting 
% for which the compressed image is similar in visual to the Processed-Original-image
imwrite(im_final,'../data/compressed_Output_JPEG_30.jpeg','jpeg','Quality', 30);
im_compressed_30 = imread('../data/compressed_Output_JPEG_30.jpeg', 'jpeg');
figure(5); subplot(1,2,1); imshow(im_final); title("Processed Original Image");
figure(5); subplot(1,2,2); imshow(im_compressed_30);
title("Compressed JPEG Image (Quality factor 30)");

% End-of-code

%% 