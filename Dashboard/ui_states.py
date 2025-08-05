import base64
import asyncio
from tqdm import tqdm

from nicegui import ui, events, run
from starlette.formparsers import MultiPartParser
#from multiprocessing import Manager, Queue
from collections import deque

MultiPartParser.spool_max_size = 1024 * 1024 * 10  # 10 MB

class GUI:
    '''
    Class to hold all the gui elements and pages
    '''
    def __init__(self, processes):
        '''
        Creates an instace of the process class for use in passing data about 
        '''
        self.processing = processes
        self.processing.output_image = None
        self.processing.defect_coords = []
        self.processing.defect_probs = []
        self.processing.flash_coords = []
        self.processing.flash_probs = []
        self.processing.ok_coords = []
        self.processing.ok_probs = []
        self.processing.scale_factor = None

        self.pages()

    def find_box_index(self, x, y, scale_factor, box_size=224):
        '''
        When a box is clicked, this finds the index of the box in the lists of coords

        Useful for displaying coordinates and probabilities
        '''
        x = x / scale_factor
        y = y / scale_factor
        if self.d_switch.value:
            for i, (top_left_x, top_left_y) in enumerate(self.processing.defect_coords):
                if (top_left_x <= x < top_left_x + box_size) and (top_left_y <= y < top_left_y + box_size):
                    box_type = 'Defect'
                    return box_type, i
        
        if self.f_switch.value:
            for i, (top_left_x, top_left_y) in enumerate(self.processing.flash_coords):
                if (top_left_x <= x < top_left_x + box_size) and (top_left_y <= y < top_left_y + box_size):
                    box_type = 'Flash'
                    return box_type, i
        
        if self.ok_switch.value:
            for i, (top_left_x, top_left_y) in enumerate(self.processing.ok_coords):
                if (top_left_x <= x < top_left_x + box_size) and (top_left_y <= y < top_left_y + box_size):
                    box_type = 'OK'
                    return box_type, i
                
        return None, None

    def pages(self):
        '''
        Holds all ui page logic 
        '''
        def on_click(click_args):
            '''
            Works out if click is inside a defect box
            If it is, then displays probability of defect at that location
            '''
            box_type, box_clicked = self.find_box_index(click_args["image_x"], click_args["image_y"], 
                                                self.processing.scale_factor)
            if box_type == 'Defect':
                coords = self.processing.defect_coords
                probs = self.processing.defect_probs
            elif box_type == 'Flash':
                coords = self.processing.flash_coords
                probs = self.processing.flash_probs
            elif box_type == 'OK':
                coords = self.processing.ok_coords
                probs = self.processing.ok_probs
            else:
                coords = None
                probs = None

            if box_clicked is not None:
                ui.notify(f'{box_type} with probability {probs[box_clicked]:.4f}', 
                          position='top', caption=f'Location: {coords[box_clicked]}')

        @ui.page('/image_upload')
        def image_upload():
            async def handle_upload(e: events.UploadEventArguments):
                ui.notify(f'Uploaded {e.name}', type="positive")
                image = base64.b64encode(e.content.read())
                await asyncio.sleep(0.5)
                # progress = ui.notification(timeout=None)
                # progress.message = "Running prediction model..."
                # progress.spinner = True
                progressbar.visible = True
                await run.io_bound(self.processing.run, image, queue)
                model_done_card.visible = True
                # progress.message = "Done!"
                # progress.type = "positive"
                # progress.spinner = False
                progressbar.visible = False
                # clear the queue when the progress bar disappears just in case
                queue.clear()
                ui.navigate.to(results)   
         
            # Create a queue to communicate with the heavy computation process
            queue = deque()
            ui.timer(0.2, callback=lambda: progressbar.set_value(queue.pop() if queue else progressbar.value))

            with ui.row().classes('w-full justify-center'):
                with ui.card():
                    with ui.column(align_items='center'):
                        ui.label('Core Defect Detector').classes('text-2xl font-semibold')
                        ui.label('Please upload core image')
                        ui.upload(on_upload=handle_upload, on_rejected=lambda: ui.notify('Rejected!')).classes('max-w-full')

                        with ui.card().classes('no-shadow') as prog_card:
                            with ui.column(align_items='center'):
                                running = ui.label('Running prediction model...')
                                progressbar = ui.linear_progress(value=0, show_value=False).bind_visibility_to(running)
                                progressbar.props('size=20px rounded color=green instant-feedback')
                                progressbar.classes('q-mt-sm')
                                progressbar.bind_visibility_to(prog_card)
                                progressbar.visible = False

                        with ui.card().classes('no-shadow bg-green') as model_done_card:
                            with ui.row().classes('items-center'):
                                ui.add_head_html('<link href="https://unpkg.com/eva-icons@1.1.3/style/eva-icons.css" rel="stylesheet" />')
                                ui.icon('eva-checkmark-circle-2', color='white').classes('text-3xl')
                                ui.label('Done!').classes('text-xl white')
                                model_done_card.visible = False

        @ui.page('/results')
        async def results():
            sf = self.processing.scale_factor
            # set up interactive image
            with ui.row().classes('w-full justify-center'):
                with ui.card().classes('flat bordered'):
                    with ui.card_section().props('horizontal'):
                        result = ui.interactive_image(self.processing.output_image)
                        with ui.card_actions().classes('w-40 q-pl-lg'):
                            with ui.column(align_items='center').classes('w-full'):
                                with ui.card().classes('no-shadow align-center'):
                                    ui.label(f'{len(self.processing.defect_coords)}').classes('w-full text-center text-4xl font-bold')
                                    ui.label('DEFECTS').classes('w-full text-center text-xl font-semibold')
                                with ui.card().classes('no-shadow align-left'):
                                    self.d_switch = ui.switch('Defects', value=True).props('color=red')
                                    self.f_switch = ui.switch('Flash').props('color=yellow')
                                    self.ok_switch = ui.switch('OK').props('color=green')
                                # button below image to go back to upload page   
                                ui.button('New image upload', on_click=lambda: ui.navigate.to(image_upload))

                        # Display red boxes at all defects found by the model
                        
                        defect_boxes = result.add_layer().bind_visibility_from(self.d_switch, 'value')
                        flash_boxes = result.add_layer().bind_visibility_from(self.f_switch, 'value')
                        ok_boxes = result.add_layer().bind_visibility_from(self.ok_switch, 'value')

                        box_d = (224 * sf) - 2

                        for defect in self.processing.defect_coords:
                            defect_boxes.content += f'''<rect x="{(defect[0] * sf)}" y="{(defect[1] * sf)}" width="{box_d}" 
                                            height="{box_d}" fill="none" stroke="red" stroke-width="2"
                                            pointer-events="none" cursor="pointer" />'''
                            
                        for flash in self.processing.flash_coords:
                            flash_boxes.content += f'''<rect x="{(flash[0]  * sf)}" y="{(flash[1] * sf)}" width="{box_d}" 
                                            height="{box_d}" fill="none" stroke="yellow" stroke-width="2"
                                            pointer-events="none" cursor="pointer" />'''
                            
                        for ok in self.processing.ok_coords:
                            ok_boxes.content += f'''<rect x="{(ok[0] * sf)}" y="{(ok[1] * sf)}" width="{box_d}" 
                                            height="{box_d}" fill="none" stroke="green" stroke-width="2"
                                            pointer-events="none" cursor="pointer" />'''
                            
                        # This is an invisible svg layer over the whole image that makes click coordinates work nicely
                        # Not the best way of doing it, but it works
                        # Click coords given by the box layers are not relative to the original image, coords given by 
                        # this svg are.    
                        result.content = f'''<rect x="0" y="0" width="1440" height="{self.processing.new_h}"
                                        fill="none" stroke="none" pointer-events="all" cursor="pointer" />'''
                            
                        # fun nicegui svg stuff, any click in a box calls the on_click function
                        result.on('svg:pointerdown', lambda e: on_click(e.args))

        # This basically starts the whole thing 
        # Could maybe be done with an on_startup callback?
        ui.navigate.to(image_upload)