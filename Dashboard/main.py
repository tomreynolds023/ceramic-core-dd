import ui_states
import model
import process

from nicegui import ui

test_model = model.Model()
model_path = 'Dashboard/RN18_opt_withAugs_weights.pth'

try:
    test_model.load_model(model_path)
except FileNotFoundError:
    raise FileNotFoundError(f"Model file '{model_path}' not found.")
except Exception as e:
    raise RuntimeError(f"Failed to load model from '{model_path}': {e}")


processes = process.Process(test_model)
gui = ui_states.GUI(processes)

if __name__ == '__main__':
    ui.run(reload=False, dark=True, title='Defect Detector', reconnect_timeout=60)
    print("Done!")