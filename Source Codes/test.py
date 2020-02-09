import argparse
from pred_waldo import detect_waldo
from pred_wenda import detect_wenda
from pred_wizard import detect_wizard


parser = argparse.ArgumentParser()
parser.add_argument('--target', type=str, choices=['all','waldo','wenda','wizard'], default='all', help='Target you wish to detect')
parser.add_argument('--visual', action='store_true', help='Enable to show detection outputs')
parser.add_argument('--loc', type=str, help='.txt file location containing images to detect', default='datasets/ImageSets/val.txt')

args = parser.parse_args()
target = args.target
visual = args.visual
loc = args.loc

if target == 'all':
	print('Performing detection for Waldo..')
	detect_waldo(visual, loc)
	print('Performing detection for Wenda..')
	detect_wenda(visual, loc)
	print('Performing detection for Wizard..')
	detect_wizard(visual, loc)

if target == 'waldo':
	print('Performing detection for Waldo..')
	detect_waldo(visual, loc)

if target == 'wenda':
	print('Performing detection for Wenda..')
	detect_wenda(visual, loc)

if target == 'wizard':
	print('Performing detection for Wizard..')
	detect_wizard(visual, loc)