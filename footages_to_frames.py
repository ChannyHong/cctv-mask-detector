# Copyright 2020 Superb AI, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Authors: Channy Hong

import os
import cv2

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--footage_dir', type=str, help='Required: the path to the footage file', default="mask_dataset/test")
parser.add_argument('--output_dir', type=str, help='Required: the path to output the annotated video file to', required=True)
parser.add_argument('--frames_per_extract', type=int, help='Required: how many frames to skip per extract', required=True)

args = parser.parse_args()

FRAMES_PER_EXTRACT = args.frames_per_extract

def main():
	footages = os.listdir(args.footage_dir)
	footages.remove(".DS_Store")

	for footage in footages:
		vidcap = cv2.VideoCapture(os.path.join(args.footage_dir, footage))
		success, image = vidcap.read()
		count = 0

		while success:
			if count % FRAMES_PER_EXTRACT == 0:
				cv2.imwrite(os.path.join(args.output_dir, '{}-frame{}.jpg'.format(footage[:-4], count)), image)
			success, image = vidcap.read()
			count += 1

# Driver code
if __name__ == "__main__":
	main()