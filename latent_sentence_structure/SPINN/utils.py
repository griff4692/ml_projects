def render_args(args):
	print ("")
	for arg in vars(args):
		print ("%s=%s" % (arg, getattr(args, arg)))
	print("")

def cudify(args, tensor):
	if args.gpu > -1:
		tensor = tensor.cuda()
	return tensor
