import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--rename_files_dir', type=int, help='Required: path to the folder that contains image files to rename ascending numerically from "1.jpeg"', required=True)
args = parser.parse_args()



def main():
	image_list = os.listdir(args.rename_files_dir)
	counter = 1

	for image in image_list:
		os.rename(os.path.join(args.rename_files_dir, image), os.path.join(args.rename_files_dir, str(counter) + ".jpeg"))
		counter += 1

# Driver code
if __name__ == "__main__":
	main()