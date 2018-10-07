# nerf_tracking


# running the test image script.
python -m scripts.label_image \
	--graph=tf_files/retrained_graph.pb \
	--image=tf_files/test_images/IMG_0267.jpg


# testing the retrained graph on the raspberry pi

# On Pi
~/AIY-projects-python/src/examples/vision/mobilenet_based_classifier.py \
  --model_path ~/retrained_graph.binaryproto \
  --label_path ~/retrained_labels.txt \
  --input_height 160 \
  --input_width 160 \
  --input_layer input \
  --output_layer final_result \
  --preview


