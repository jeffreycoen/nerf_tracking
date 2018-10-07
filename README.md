# nerf_tracking


# running the test image script.
python -m scripts.label_image \
	--graph=tf_files/retrained_graph.pb \
	--image=tf_files/test_images/IMG_0267.jpg


# testing the retrained graph on the raspberry pi

# On Pi
./test_run_model_on_bonnet.py  \
  --model_path ~/AIY-projects-python/src/aiy/vision/models/retrained_graph.binaryproto \
  --input_height 160   \
  --input_width 160 


## Run the model!!
./mobilenet_based_classifier.py  \
  --model_path ~/AIY-projects-python/src/aiy/vision/models/retrained_graph.binaryproto \
  --label_path ~/AIY-projects-python/src/aiy/vision/models/retrained_labels.txt \
  --input_height 160   \
  --input_width 160   \
  --input_layer input   \
  --output_layer final_result   \
  --threshold 0.6   \
  --preview   \
  --show_fps
