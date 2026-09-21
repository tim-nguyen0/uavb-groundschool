## Image Stitcher

For this image stitcher, I wanted to challenge myself and hand roll most of the code to align and stitch images. In CS180, we recently build an image aligner from first principles using NCC and I thought I'd implement some of that code to apply here. 

Given that the video we are stitching together is pretty much exclusively linear motion, cartesian alignment is enough.

Visit [my CS180 project 1 website](https://tim-nguyen0.github.io/cs180/1/p1.html) to see the implementation for all of the alignment apparatus.

Essentially, to align 2 images, I normalize both (grayscaled) images by removing their bias and dividing by $\sqrt{variance}$ with respect to intensities, correlate them componentwise, and maximize the average correlations per pixel. 

There are some other steps taken to ensure better edge detections for lots of (band passes), as well as image pyramiding for performance (although pyramidding also notably reduces the quality of results for noisy areas).

For this project, I sample images at some constant sampling rate and align and overlay samples sequentially after calculating offsets.

[image-ref]: ./final_images/stitch0.5hz.png "Image"
