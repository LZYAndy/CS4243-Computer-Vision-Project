import argparse
from svm import SVM


parser = argparse.ArgumentParser()
parser.add_argument('--target', type=str, choices=['waldo','wenda','wizard'], required=True, help='Target you wish to detect')

args = parser.parse_args()
target = args.target

if target == 'waldo':
	print('Performing detection for Waldo..')
	svm = SVM('waldo', 10, 8)
	svm.train()

if target == 'wenda':
	print('Performing detection for Wenda..')
	svm = SVM('wenda')
	svm.train()

if target == 'wizard':
	print('Performing detection for Wizard..')
	detect_wizard(visual)
	svm = SVM('wizard')
	svm.train()
