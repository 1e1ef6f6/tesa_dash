-- PostgreSQL Schema for TESA Dashboard
-- Converted from db.sql diagram

-- Table: map
CREATE TABLE IF NOT EXISTS map (
    id SERIAL PRIMARY KEY,
    lat FLOAT,
    lon FLOAT,
    alt FLOAT,
    time_map TIMESTAMP,
    source VARCHAR
);

-- Table: drone
CREATE TABLE IF NOT EXISTS drone (
    id VARCHAR PRIMARY KEY,
    type VARCHAR,
    weight FLOAT,
    team VARCHAR, -- ระบุทีมของโดรน (offense หรือ defense)
    tem FLOAT,
    hum FLOAT,
    wind_speed FLOAT,
    direction FLOAT,
    battery FLOAT,
    cpu_tem FLOAT
);

-- Table: map_atk
CREATE TABLE IF NOT EXISTS map_atk (
    uid SERIAL PRIMARY KEY,
    drone_id VARCHAR,
    angular FLOAT,
    speed FLOAT,
    map_id INTEGER,
    time_atk TIMESTAMP,
    team VARCHAR, -- ฝั่งบุก (offense) หรือ (defense)
    CONSTRAINT fk_map_atk_map FOREIGN KEY (map_id) REFERENCES map(id),
    CONSTRAINT fk_map_atk_drone FOREIGN KEY (drone_id) REFERENCES drone(id)
);

-- Table: def_cam
CREATE TABLE IF NOT EXISTS def_cam (
    id SERIAL PRIMARY KEY,
    img_path TEXT,
    time_def TIMESTAMP,
    type_drone VARCHAR,
    size FLOAT,
    confidence FLOAT,
    map_id INTEGER,
    atk_id INTEGER, -- เชื่อมโยงการโจมตี
    drone_id VARCHAR, -- เชื่อมโยงกับโดรน
    is_enemy BOOLEAN, -- ระบุว่าโดรนที่ตรวจจับเป็นศัตรูหรือไม่
    enemy_team VARCHAR, -- ทีมของโดรนศัตรู (offense หรือ defense)
    CONSTRAINT fk_def_cam_map FOREIGN KEY (map_id) REFERENCES map(id),
    CONSTRAINT fk_def_cam_atk FOREIGN KEY (atk_id) REFERENCES map_atk(uid),
    CONSTRAINT fk_def_cam_drone FOREIGN KEY (drone_id) REFERENCES drone(id)
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_map_atk_map_id ON map_atk(map_id);
CREATE INDEX IF NOT EXISTS idx_map_atk_drone_id ON map_atk(drone_id);
CREATE INDEX IF NOT EXISTS idx_map_atk_time_atk ON map_atk(time_atk);
CREATE INDEX IF NOT EXISTS idx_def_cam_map_id ON def_cam(map_id);
CREATE INDEX IF NOT EXISTS idx_def_cam_atk_id ON def_cam(atk_id);
CREATE INDEX IF NOT EXISTS idx_def_cam_drone_id ON def_cam(drone_id);
CREATE INDEX IF NOT EXISTS idx_def_cam_time_def ON def_cam(time_def);

