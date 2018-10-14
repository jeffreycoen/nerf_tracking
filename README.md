# nerf_tracking

# start the virtual environment
cd ml
source env/bin/activate

#building the model
python -m scripts.retrain \
  --bottleneck_dir=tf_files/bottlenecks \
  --how_many_training_steps=4000 \
  --model_dir=tf_files/models/ \
  --summaries_dir=tf_files/training_summaries/"mobilenet_0.50_160" \
  --output_graph=tf_files/retrained_graph.pb \
  --output_labels=tf_files/retrained_labels.txt \
  --architecture="mobilenet_0.50_160" \
  --image_dir=tf_files/flower_photos   \
  --random_brightness 15

# running the test image script.
python -m scripts.label_image \
	--graph=tf_files/retrained_graph.pb \
	--image=tf_files/test_images/IMG_0267.jpg


# compile that graph!
./bonnet_model_compiler.par\
  --frozen_graph_path=retrained_graph.pb\
  --output_graph_path=retrained_graph.binaryproto\
  --input_tensor_name=input\
  --output_tensor_names=final_result\
  --input_tensor_size=160 

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
  --threshold 0.8   \
  --preview   \
  --show_fps
