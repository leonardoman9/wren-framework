#!/usr/bin/env python3
"""
Test BirdNET analyzer with different configurations
"""

def test_analyzer_formats():
    print("🔧 TESTING BIRDNET ANALYZER CONFIGURATIONS")
    print("=" * 60)
    
    import tempfile
    import os
    
    # Test different species list formats
    formats_to_test = [
        ("Format 1: Scientific only", "Corvus corax"),
        ("Format 2: Scientific_Common", "Corvus corax_Common Raven"),
        ("Format 3: Common only", "Common Raven"),
    ]
    
    for format_name, species_line in formats_to_test:
        print(f"\n🧪 {format_name}")
        
        # Create temp species file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(f"{species_line}\n")
            temp_file = f.name
        
        try:
            from birdnetlib.analyzer import Analyzer
            from birdnetlib import Recording
            
            # Test analyzer creation
            analyzer = Analyzer(custom_species_list_path=temp_file)
            print(f"   ✅ Analyzer created successfully")
            
            # Test on a real file
            import glob
            test_files = glob.glob("bird_sound_dataset_processed/Corvus_corax/*.wav")[:1]
            
            if test_files:
                test_file = test_files[0]
                recording = Recording(analyzer, test_file, min_conf=0.001)
                recording.analyze()
                
                print(f"   📊 Detections: {len(recording.detections)}")
                if recording.detections:
                    for det in recording.detections[:3]:
                        print(f"      - {det['scientific_name']} ({det['confidence']:.4f})")
                else:
                    print("   ❌ No detections")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
        
        finally:
            os.unlink(temp_file)
    
    # Test without custom species list
    print(f"\n🧪 No custom species list (default)")
    try:
        analyzer = Analyzer()
        print(f"   ✅ Default analyzer created")
        
        if test_files:
            recording = Recording(analyzer, test_files[0], min_conf=0.001)
            recording.analyze()
            
            print(f"   📊 Detections: {len(recording.detections)}")
            corvus_found = [d for d in recording.detections if 'corvus' in d['scientific_name'].lower()]
            if corvus_found:
                print(f"   🎯 Corvus detections: {len(corvus_found)}")
                for det in corvus_found[:3]:
                    print(f"      - {det['scientific_name']} ({det['confidence']:.4f})")
            else:
                print("   ❌ No Corvus detections in default mode")
                
    except Exception as e:
        print(f"   ❌ Default analyzer failed: {e}")

if __name__ == "__main__":
    test_analyzer_formats()