import matplotlib.pyplot as plt
from PIL import Image
import os
import os.path as osp
import numpy as np
import json

# sort_by options: 'combined', 'id' or 'landmarks'
def extract_ordered_anon_image_names(eval_dict, sort_by='combined'):
    
    image_losses = []   

    for image_entry in eval_dict['image_eval']:
        image_name = image_entry['image_name']
        if sort_by == 'combined':
            final_chosen_loss = image_entry['combined_losses'][-1]['loss']
        elif sort_by == 'id':
            final_chosen_loss = image_entry['id_loss'][-1]['loss']
        elif sort_by == 'landmarks':
            final_chosen_loss = image_entry['lm_loss'][-1]['loss']
        image_losses.append({'image_name': image_name, 'loss': final_chosen_loss})
            
    sorted_images = sorted(image_losses, key=lambda x: x['loss'])
            
    return sorted_images

def compare_images(real_images_dir, anon_dir, show_top_x, show_bottom_x, save=True):
    
    with open(osp.join(anon_dir, 'eval.json'), 'r') as file:
        eval_dict = json.load(file)
    
    sorted_images = extract_ordered_anon_image_names(eval_dict)
    if show_top_x > 0 :
        top_x_images = np.array(sorted_images[:show_top_x])
        images_to_display = top_x_images
    if show_bottom_x > 0:
        bottom_x_images = np.flip(np.array(sorted_images[-show_bottom_x:]))
        images_to_display = bottom_x_images
    if show_bottom_x > 0 and show_top_x > 0:
        images_to_display = np.concatenate((top_x_images, bottom_x_images))
    
    num_rows = show_top_x + show_bottom_x
    
    fig_size_multi = 4
    
    fig, axs = plt.subplots(num_rows, 2, figsize=(fig_size_multi * 2, fig_size_multi * num_rows), gridspec_kw={'wspace': 0, 'hspace': 0})
    
    for row_index, image in enumerate(images_to_display):
        real_image = Image.open(osp.join(real_images_dir, f"{image['image_name']}.jpg"))
        anon_image = Image.open(osp.join(anon_dir, 'data', f"{image['image_name']}.jpg"))
        
        ax = axs[row_index][0]
        ax.imshow(real_image)
        ax.axis('off')
        title_text = f"top: {row_index+1}" if row_index < show_top_x else f"bottom {row_index+1 - show_top_x}"
        ax.text(0.5, 0.05, f"Image Ranked {title_text}", fontsize=12, color='white', 
            ha='center', va='bottom', transform=ax.transAxes, bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.5'))
        
        ax = axs[row_index][1]
        ax.imshow(anon_image)
        ax.axis('off')
        ax.text(0.5, 0.05, f"Image Loss: {image['loss']}", fontsize=12, color='white',
            ha='center', va='bottom', transform=ax.transAxes, bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.5'))
    
    fig.text(0.25, 1, 'Real Images', ha='center', va='center', fontsize=16, color='white', bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.5'))
    fig.text(0.75, 1, 'Anonymized Images', ha='center', va='center', fontsize=16, color='white', bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.5'))
    
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    
    plt.tight_layout(pad=0)
    if save:
        plt.savefig(f"{anon_dir}_top_{show_top_x}_bottom_{show_bottom_x}.png", bbox_inches='tight', dpi=300)
    plt.show()
        
        
def graph_data(anon_root_dir, loss_type, criteria):
    all_anon_dirs = os.listdir(anon_root_dir)
    
    criteria_indeces = ['', 'id_margin', 'lambda_id', 'lambda_attr', 'optimizer', 'lr', '', '', '', 'nn_type']
    
    criteria_entries = []
    data_to_graph = []
    for anon_dir in all_anon_dirs:
        criteria_entries.append(anon_dir.split("-")[criteria_indeces.index(criteria)])
        
    
    unique_crtieria_vals = list(dict.fromkeys(criteria_entries))
    for opt in unique_crtieria_vals:
        data_to_graph.append({criteria: opt, 'losses': []})
        


    for anon_dir in all_anon_dirs:        
        criteria_val = anon_dir.split("-")[criteria_indeces.index(criteria)]
        
        with open(osp.join(anon_root_dir, anon_dir, 'eval.json'), 'r') as file:
            eval_dict = json.load(file)
            
            
        losses = []
        for val in eval_dict['image_eval'][0]['combined_losses']:
            losses.append({'epoch': val['epoch'], 'losses': []})

        for image_entry in eval_dict['image_eval']:
            if loss_type == 'combined':
                for index, loss in enumerate(image_entry['combined_losses']):
                    losses[index]['losses'].append(loss['loss'])
            elif loss_type == 'id':
                for index, loss in enumerate(image_entry['id_loss']):
                    losses[index]['losses'].append(loss['loss'])
            elif loss_type == 'landmarks':
                for index, loss in enumerate(image_entry['lm_loss']):
                    losses[index]['losses'].append(loss['loss'])

        for data in data_to_graph:
            if data[criteria] == criteria_val:
                if len(data['losses']) == 0:
                    data['losses'] = losses
                else:
                    for l_index, epoch_losses in enumerate(losses):
                        data['losses'][l_index]['losses'].extend(epoch_losses['losses'])
                        
    for data in data_to_graph:
        for epoch_losses in data['losses']:
            epoch_losses['average'] = np.average(epoch_losses['losses'])
            del epoch_losses['losses']
    
    plt.figure(figsize=(10, 6))

    for data in data_to_graph:
        criteria_name = data[criteria]
        epochs = [entry['epoch'] for entry in data['losses']]
        averages = [entry['average'] for entry in data['losses']]
        
        plt.plot(epochs, averages, label=f'{criteria}: {criteria_name}', marker='o')

    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title(f"Average Loss per Epoch for {criteria}")
    plt.legend()
    plt.grid(True)
    plt.show()
        

