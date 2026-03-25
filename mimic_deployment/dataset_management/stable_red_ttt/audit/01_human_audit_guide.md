| Episode range | Current label | Viewer link (first episode) |
|---|---|---|
| ep_0_39 | pick red x handover place center | https://huggingface.co/spaces/lerobot/visualize_dataset?dataset=Mimic-Robotics%2Ffull_ttt_redx_stable&episode=0 |
| ep_40_74 | pick red x handover place top left | https://huggingface.co/spaces/lerobot/visualize_dataset?dataset=Mimic-Robotics%2Ffull_ttt_redx_stable&episode=40 |
| ep_75_124 | pick red x handover place top right | https://huggingface.co/spaces/lerobot/visualize_dataset?dataset=Mimic-Robotics%2Ffull_ttt_redx_stable&episode=75 |
| ep_125_164 | pick red x handover place bottom left | https://huggingface.co/spaces/lerobot/visualize_dataset?dataset=Mimic-Robotics%2Ffull_ttt_redx_stable&episode=125 |
| ep_165_204 | pick red x handover place bottom right | https://huggingface.co/spaces/lerobot/visualize_dataset?dataset=Mimic-Robotics%2Ffull_ttt_redx_stable&episode=165 |
| ep_205_223 | pick red x handover place top middle | https://huggingface.co/spaces/lerobot/visualize_dataset?dataset=Mimic-Robotics%2Ffull_ttt_redx_stable&episode=205 |
| ep_224_259 | pick red x handover place middle left | https://huggingface.co/spaces/lerobot/visualize_dataset?dataset=Mimic-Robotics%2Ffull_ttt_redx_stable&episode=224 |
| ep_260_299 | pick red x handover place middle right | https://huggingface.co/spaces/lerobot/visualize_dataset?dataset=Mimic-Robotics%2Ffull_ttt_redx_stable&episode=260 |
| ep_300_339 | pick red x handover place bottom middle | https://huggingface.co/spaces/lerobot/visualize_dataset?dataset=Mimic-Robotics%2Ffull_ttt_redx_stable&episode=300 |
| ep_340_379 | pick red x handover place bottom left | https://huggingface.co/spaces/lerobot/visualize_dataset?dataset=Mimic-Robotics%2Ffull_ttt_redx_stable&episode=340 |
| ep_380_419 | pick red x handover place bottom middle | https://huggingface.co/spaces/lerobot/visualize_dataset?dataset=Mimic-Robotics%2Ffull_ttt_redx_stable&episode=380 |
| ep_420_459 | pick red x handover place bottom right | https://huggingface.co/spaces/lerobot/visualize_dataset?dataset=Mimic-Robotics%2Ffull_ttt_redx_stable&episode=420 |
| ep_460_501 | pick red x handover place top left | https://huggingface.co/spaces/lerobot/visualize_dataset?dataset=Mimic-Robotics%2Ffull_ttt_redx_stable&episode=460 |
| ep_502_541 | pick red x handover place middle left | https://huggingface.co/spaces/lerobot/visualize_dataset?dataset=Mimic-Robotics%2Ffull_ttt_redx_stable&episode=502 |
| ep_542_561 | pick red x handover place center | https://huggingface.co/spaces/lerobot/visualize_dataset?dataset=Mimic-Robotics%2Ffull_ttt_redx_stable&episode=542 |
| ep_562_601 | pick red x handover place middle right | https://huggingface.co/spaces/lerobot/visualize_dataset?dataset=Mimic-Robotics%2Ffull_ttt_redx_stable&episode=562 |
| ep_602_637 | pick red x handover place top middle | https://huggingface.co/spaces/lerobot/visualize_dataset?dataset=Mimic-Robotics%2Ffull_ttt_redx_stable&episode=602 |
| ep_638_677 | pick red x handover place top right | https://huggingface.co/spaces/lerobot/visualize_dataset?dataset=Mimic-Robotics%2Ffull_ttt_redx_stable&episode=638 |

```python
# Fill in what the robot is ACTUALLY doing in each block.
# Use the exact position strings: top_left, top_middle, top_right,
#                                  middle_left, middle_middle, middle_right,
#                                  bottom_left, bottom_middle, bottom_right
audit = {
    "ep_0_39":   "middle_middle", 
    "ep_40_74":  "top_left",
    "ep_75_124": "top_right",
    "ep_125_164": "bottom_left",
    "ep_165_204": "bottom_right",
    "ep_205_223": "top_middle",
    "ep_224_259": "middle_left",
    "ep_260_299": "middle_right",
    "ep_300_339": "bottom_middle",
    "ep_340_379": "bottom_left",
    "ep_380_419": "bottom_middle",
    "ep_420_459": "bottom_right",
    "ep_460_501": "top_left",
    "ep_502_541": "middle_left",
    "ep_542_561": "middle_middle",
    "ep_562_601": "middle_right",
    "ep_602_637": "top_middle",
    "ep_638_677": "top_right",
}
```