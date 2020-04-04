from saliency.fullgrad import FullGrad

def main():
	model = resnet18(pretrained=True)
    model = model.to(device)
    
	# Initialize FullGrad object
	# see below for model specs
	fullgrad = FullGrad(model)

	# Check completeness property
	# done automatically while initializing object
	fullgrad.checkCompleteness()

	# Obtain fullgradient decomposition
	input_gradient_term, bias_gradient_term = fullgrad.fullGradientDecompose(input_image, target_class)

	# Obtain saliency maps
	saliency_map = fullgrad.saliency(input_image, target_class)

if __name__ == '__main__':
    main()