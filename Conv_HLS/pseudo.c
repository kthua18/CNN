Psuedocode: 

for (int lyr = 0; lyr < NUM_LAYERS; lyr++) {
	for (int img = 0; img < size_out[lyr]; img++) {
		for (int kernel = 0; kernel < size_in[lyr]; kernel++) {
			for (int ir; ir < IMG_ROW; ir++) {
				for (int ic; ic < IMG_COL; ic++) {
					for (int kr = 0; kr < kernel_r[lyr]; kr++) {
						for (int kc = 0; kc < kernel_c[lyr]; kc++) {
							output[lyr][img][kernel][ir][ic] += 
									convolve_kernel(lyr, img, kernel, ir, ic, kr, kc);
						}
					}
				}
			}
		}
		output[lyr][img][:][:] += bias[lyr][img];
	}
	if (lyr < NUM_LAYERS - 1) {
		output[lyr][:][:][:] += relu(output[lyr][:][:][:]);
	} else {
		output[lyr][:][:][:] += sigmoid(output[lyr][:][:][:]);
	}
}

int sum = 0;
for (int i = 0; i < 10; i+=2) {
	sum += a[i];
	sum += a[i+1];
}







