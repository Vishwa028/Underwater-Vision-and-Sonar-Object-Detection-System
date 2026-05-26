# Desktop Version

This project now includes a standalone desktop application so the sensor, camera, and prediction loop can run without Streamlit.

## Run

```powershell
python standalone_app.py
```

## What It Does

- Reads `Distance:<value>` packets from the Arduino Uno.
- Captures live video from the default camera.
- Uses a faster local desktop UI instead of Streamlit reruns.
- Uses a lightweight MobileNet-SSD vision model to label real objects in the live feed.
- Adds underwater-ready image enhancement modes for blue/green cast, low light, and murky scenes.
- Sends `H` for danger and `S` for safe back to the Arduino.
- Logs state changes to `logs/events.csv`.
- Can load a custom marine SLM from `models/marine_classifier.keras` when available.

## Current Notes

- The current desktop app keeps compatibility with the existing Arduino sketch.
- `LIVE ARDUINO` reads sensor values from serial.
- `MANUAL SIMULATION` lets you test the interface without hardware.
- The detection logic now combines object labels from the camera feed with the current distance reading.
- `UNDERWATER RESEARCH` mode boosts contrast, sharpness, and color balance before detection.
- `LOW LIGHT WATER` mode adds denoising and extra brightening for darker scenes.
- The dashboard now shows water clarity state and a tunable detection threshold.
- If you train the marine classifier, the app will show `SLM: READY` in the pipeline status.
- Vision model files are stored in `models/`.

## Marine SLM Starter

Install the ML dependency:

```powershell
pip install -r requirements-ml.txt
```

Capture or place images in the dataset folders described in [dataset/README.md](/C:/Users/HP/OneDrive/Desktop/My%20web/project/sonar_project/dataset/README.md), then train:

```powershell
python prepare_marine_dataset.py "C:\path\to\downloaded\fish_images" fish
python train_marine_classifier.py --epochs 12
```

After training, restart:

```powershell
python standalone_app.py
```

The desktop app will automatically use:

- `models/marine_classifier.keras`
- `models/marine_labels.json`

## Next Upgrades

- Richer Arduino packets such as `DIST`, `AVG`, and `STATE`
- Buzzer and servo control
- Custom-trained camera-based classification for your project-specific objects
- Packaging to Windows `.exe`
