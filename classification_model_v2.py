import os 

 

def rename_files(start_number, directory_path, extension=".jpg"): 

    # Få en liste over alle filer i mappen 

    files = os.listdir(directory_path) 

    print(f"Alle filer i mappen: {files}")  # Feilsøking: Print alle filene 

 

    # Filtrer ut de som har ønsket utvidelse 

    relevant_files = [f for f in files if f.endswith(extension)] 

    print(f"Relevante filer som vil bli omdøpt: {relevant_files}")  # Feilsøking: Print relevante filer 

 

    # Sorter listen 

    relevant_files.sort() 

 

    # Omdøp filene 

    for i, filename in enumerate(relevant_files, start=start_number): 

        new_name = f"Terrestrial_{i+268}{extension}" 

        old_path = os.path.join(directory_path, filename) 

        new_path = os.path.join(directory_path, new_name) 

 

        # Endre navn på filen 

        print(f"Omdøper {old_path} til {new_name}")  # Feilsøking: Print handlingen som skal utføres 

        os.rename(old_path, new_path) 

 

# Bruksområde 

start_number = 0  # Startnummeret for omdøping 

directory_path = r'C:\Users\adriva001\Downloads\T'  # Riktig sti til mappen med filene 

rename_files(start_number, directory_path) 

 
