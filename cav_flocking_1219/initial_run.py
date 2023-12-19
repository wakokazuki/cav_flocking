"""GitHubからクローンした後に最初に実行"""

import os

from path import path_straight

def main():
    """outputのディレクトリを作成
    """
    # output
    output_path = "out"
    csv_path = os.path.join(output_path, "csv")
    image_path = os.path.join(output_path, "img")
    video_path = os.path.join(output_path, "video")
    convergence_path = os.path.join(output_path, "convergence") 
    mixed_path = os.path.join(output_path, "mixed")

    os.makedirs(output_path, exist_ok=True)
    os.makedirs(csv_path, exist_ok=True)
    os.makedirs(image_path, exist_ok=True)
    os.makedirs(video_path, exist_ok=True)
    os.makedirs(convergence_path, exist_ok=True)

    # image 
    vel_path = os.path.join(image_path, "velocity")
    accel_path = os.path.join(image_path, "accel")
    initial_position_path = os.path.join(image_path, "init")
    agent_route_path = os.path.join(image_path, "route")
    angle_path = os.path.join(image_path, "angle")
    frame_path = os.path.join(image_path, "frame")

    os.makedirs(vel_path, exist_ok=True)
    os.makedirs(accel_path, exist_ok=True)
    os.makedirs(initial_position_path, exist_ok=True)
    os.makedirs(agent_route_path, exist_ok=True)
    os.makedirs(angle_path, exist_ok=True)
    os.makedirs(frame_path, exist_ok=True)

    # leader 
    leader_setup_path = os.path.join(output_path, "leader")
    leader_csv_path = os.path.join(leader_setup_path, "csv")
    leader_img_path = os.path.join(leader_setup_path, "img")
    leader_video_path = os.path.join(leader_setup_path, "video")

    os.makedirs(leader_setup_path, exist_ok=True)
    os.makedirs(leader_csv_path, exist_ok=True)
    os.makedirs(leader_img_path, exist_ok=True)
    os.makedirs(leader_video_path, exist_ok=True)

    # heatmap
    heatmap_path = "out/heatmap"
    heatmap_csv_path = os.path.join(heatmap_path, "csv")
    heatmap_img_path = os.path.join(heatmap_path, "img")

    os.makedirs(heatmap_path, exist_ok=True)
    os.makedirs(heatmap_csv_path, exist_ok=True) 
    os.makedirs(heatmap_img_path, exist_ok=True) 

    # convergence
    convergence_csv_path = os.path.join(convergence_path, "csv")
    convergence_img_path = os.path.join(convergence_path, "img")
    
    os.makedirs(convergence_csv_path, exist_ok=True)
    os.makedirs(convergence_img_path, exist_ok=True)

    # mixed traffic
    mixed_csv_path = os.path.join(mixed_path, "csv")
    mixed_img_path = os.path.join(mixed_path, "img")

    os.makedirs(mixed_csv_path, exist_ok=True)
    os.makedirs(mixed_img_path, exist_ok=True)


if __name__ == "__main__":
    main()
    # リーダーの経路を作成
    max_vel_list = [4.0, 6.0, 8.0, 10.0, 12.0] # [m/s] 最大速度のリスト
    for i in range(len(max_vel_list)):
        path_straight.main(show_flag=False, max_vel=max_vel_list[i])
        
    
    
