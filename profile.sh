# Run a profile of the input file, save it to out.prof, and display it with snakeviz

# Usage: profile.sh input_file.py

# Run the profile
python -m cProfile -o out.prof $1

# Display the profile
snakeviz out.prof