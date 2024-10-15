python generate_reasoning.py
echo "-----------------------------------------------------------"
echo "Completed Generation for Train"
echo "------------------------------------------------------------"
python generate_reasoning_valid.py
echo "-----------------------------------------------------------"
echo "Completed Generation for Valid"
echo "------------------------------------------------------------"
python generate_reasoning_test.py
echo "-----------------------------------------------------------"
echo "Completed Generation for Test"
echo "------------------------------------------------------------"