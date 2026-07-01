import random
import numpy as np
from collections import Counter
from typing import List, Dict, Set, Tuple
from tqdm import tqdm

class StratifiedSplitter:
    def __init__(self, seed: int = 42):
        self.seed = seed
        
    def split(
        self,
        img_to_classes: Dict[str, Set[int]],
        class_to_imgs: Dict[int, List[str]],
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
        min_samples: int = 2,
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Perform stratified split.
        """
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        # Validate ratios
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Split ratios must sum to 1.0"
        
        all_stems = list(img_to_classes.keys())
        random.shuffle(all_stems)
        
        train_stems, val_stems, test_stems = [], [], []
        train_class_counts, val_class_counts, test_class_counts = Counter(), Counter(), Counter()
        
        n_total = len(all_stems)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        print(f"\nTarget split sizes: Train: {n_train}, Val: {n_val}, Test: {n_total - n_train - n_val}")
        
        assigned = set()
        
        # First pass: Rare classes
        print("\n📊 First pass: ensuring minimum samples for rare classes...")
        for cls, imgs in sorted(class_to_imgs.items(), key=lambda x: len(x[1])):
            class_imgs = [img for img in imgs if img not in assigned]
            if len(class_imgs) < min_samples * 3:
                # Distribute rare class
                random.shuffle(class_imgs)
                if len(class_imgs) >= 3:
                    splits = [train_stems, val_stems, test_stems]
                    split_counts = [train_class_counts, val_class_counts, test_class_counts]
                    
                    for i in range(3):
                        stem = class_imgs[i]
                        splits[i].append(stem)
                        assigned.add(stem)
                        for c in img_to_classes[stem]:
                            split_counts[i][c] += 1
        
        # Second pass: Remaining images
        print("\n📊 Second pass: distributing remaining images...")
        remaining = [stem for stem in all_stems if stem not in assigned]
        random.shuffle(remaining)
        
        for stem in tqdm(remaining, desc="Assigning images"):
            current_train = len(train_stems)
            current_val = len(val_stems)
            current_test = len(test_stems)
            
            train_need = (n_train - current_train) / n_train if n_train > 0 else 0
            val_need = (n_val - current_val) / n_val if n_val > 0 else 0
            test_target = n_total - n_train - n_val
            test_need = (test_target - current_test) / test_target if test_target > 0 else 0
            
            total_need = train_need + val_need + test_need
            if total_need > 0:
                train_prob = train_need / total_need
                val_prob = val_need / total_need
            else:
                train_prob, val_prob = train_ratio, val_ratio
                
            rand = random.random()
            if rand < train_prob:
                train_stems.append(stem)
            elif rand < train_prob + val_prob:
                val_stems.append(stem)
            else:
                test_stems.append(stem)
        
        return train_stems, val_stems, test_stems
