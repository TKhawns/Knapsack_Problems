import os
import random
import numpy as np

def generate_weights(widths, heights, profits):
    """
    Generate weights for rectangles based on their profits with controlled randomness.
    
    Parameters:
    - widths: List of rectangle widths
    - heights: List of rectangle heights
    - profits: List of rectangle profits
    
    Returns:
    - weights: List of generated weights
    - capacity: Total weight capacity constraint
    """
    # Calculate areas (still useful for some randomization)
    areas = [w * h for w, h in zip(widths, heights)]
    
    # Base weight calculation: primarily profit-based with small area influence
    max_profit = max(profits) if profits and max(profits) > 0 else 1
    
    # Generate weights with controlled randomness
    # Base weight = profit × (0.8 + random factor between 0 and 0.4)
    weights = []
    for profit, area in zip(profits, areas):
        # Small random variation (±20% of profit)
        random_factor = 0.8 + random.uniform(0, 0.4)
        # Small area influence (5-15% based on normalized area)
        area_factor = 0.05 + (0.1 * area / max(areas)) if max(areas) > 0 else 0.1
        
        # Weight is primarily based on profit with small variations
        weight = int(max(1, round(profit * random_factor * (1 + area_factor))))
        weights.append(weight)
    
    # Set capacity bound as a percentage of total weight
    # Use a tighter bound for larger instances to make the problem challenging
    total_weight = sum(weights)
    n = len(weights)
    
    # Adjust capacity percentage based on instance size
    if n <= 10:
        capacity_percentage = random.uniform(0.65, 0.75)  # 65-75% for small instances
    elif n <= 20:
        capacity_percentage = random.uniform(0.60, 0.70)  # 60-70% for medium instances
    else:
        capacity_percentage = random.uniform(0.55, 0.65)  # 55-65% for large instances
    
    # Ensure capacity allows at least the most valuable items to be packed
    # Sort items by profit/weight ratio (efficiency)
    efficiency = [(p/w, p, w, i) for i, (p, w) in enumerate(zip(profits, weights))]
    efficiency.sort(reverse=True)
    
    # Calculate minimum capacity to ensure the problem is feasible
    min_capacity = 0
    min_profit_sum = sum(profits) * 0.4  # Ensure at least 40% of total profit is achievable
    current_profit = 0
    
    for _, profit, weight, _ in efficiency:
        if current_profit < min_profit_sum:
            min_capacity += weight
            current_profit += profit
        else:
            break
    
    # Final capacity is max of calculated percentage and minimum feasible capacity
    capacity = max(int(total_weight * capacity_percentage), min_capacity)
    
    # Ensure capacity is at least 10% more than the weight of the heaviest item
    capacity = max(capacity, int(max(weights) * 1.1))
    
    return weights, capacity

def process_dataset_folder(input_folder_path, output_folder_path):
    """Process all instance files in the given folder and save to a new folder."""
    print(f"Processing files in {input_folder_path}")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
    
    for filename in os.listdir(input_folder_path):
        if not filename.endswith('.txt'):
            continue
            
        input_file_path = os.path.join(input_folder_path, filename)
        output_file_path = os.path.join(output_folder_path, filename)
        print(f"Processing {filename}...")
        
        try:
            with open(input_file_path, 'r') as f:
                lines = [line.strip() for line in f.readlines() if not line.startswith('#')]
                
            # Parse the existing data
            strip_dims = list(map(int, lines[0].split()))
            widths = list(map(int, lines[1].split()))
            heights = list(map(int, lines[2].split()))
            
            # Handle cases where profit line might be missing or empty
            profits = []
            if len(lines) > 3 and lines[3].strip():
                try:
                    profits = list(map(int, lines[3].split()))
                except ValueError:
                    # If profit line can't be parsed, generate default profits
                    print(f"Warning: Could not parse profit line in {filename}, generating default profits")
                    profits = [w * h for w, h in zip(widths, heights)]
            else:
                # If profit line is missing, generate default profits based on area
                print(f"Warning: Missing profit line in {filename}, generating default profits")
                profits = [w * h for w, h in zip(widths, heights)]
            
            # Ensure profits list is the same length as widths/heights
            if len(profits) != len(widths):
                print(f"Warning: Profit list length mismatch in {filename}, adjusting")
                if len(profits) < len(widths):
                    # Extend profits if needed
                    profits.extend([w * h for w, h in zip(widths[len(profits):], heights[len(profits):])])
                else:
                    # Truncate profits if too long
                    profits = profits[:len(widths)]
            
            # Generate weights and capacity
            weights, capacity = generate_weights(widths, heights, profits)
            
            # Prepare the updated content with capacity in the first line
            # Format: W H C
            updated_content = [
                f"{strip_dims[0]} {strip_dims[1]} {capacity}\n",  # W H C
                ' '.join(map(str, widths)) + '\n',  # widths
                ' '.join(map(str, heights)) + '\n',  # heights
                ' '.join(map(str, profits)) + '\n',  # profits
                ' '.join(map(str, weights)) + '\n',  # weights
            ]
            
            # Write to the new file
            with open(output_file_path, 'w') as f:
                f.writelines(updated_content)
                
            print(f"Created {filename} with weights and capacity bound in output folder")
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    input_dataset_folder = "/Users/khanhtrinh/Desktop/Workspace/Intern_ESLab/SAT_Problems/KLTN/KP_WithoutWeight/dataset/all_data"
    output_dataset_folder = "/Users/khanhtrinh/Desktop/Workspace/Intern_ESLab/SAT_Problems/KLTN/KP_WithoutWeight/dataset/all_data_weight"
    process_dataset_folder(input_dataset_folder, output_dataset_folder)
    print("Weight generation complete!")