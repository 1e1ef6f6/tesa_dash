// global axios, io, mapboxgl
const App = (() => {
  let CONFIG = null;
  let socket = null;
  const MAPBOX_TOKEN = 'pk.eyJ1IjoiY2hhdGNoYWxlcm0iLCJhIjoiY21nZnpiYzU3MGRzdTJrczlkd3RxamN4YyJ9.k288gnCNLdLgczawiB79gQ'; // ⚠️ ใช้ Token จริงของคุณ

  const state = {
    def: { markers: {}, expanded: false, map: null }, // markers ใช้สำหรับเก็บ Mapbox Marker
    off: { markers: {}, expanded: false, map: null }
  };

  // -------- utils --------
  async function loadConfig() {
    // แก้ไข: ใช้ '/config' แทนเพื่อให้เรียกจาก Proxy
    const { data } = await axios.get('/config', { timeout: 8000 });
    CONFIG = data;
    if (!data.defence.set || !data.offence.set) {
      console.warn("Camera ENV not set", data);
    }
    return data;
  }
  function setBadge(el, ok) {
    el.classList.toggle('connected', ok);
    el.classList.toggle('disconnected', !ok);
    el.textContent = ok ? 'Connected' : 'Disconnected';
  }
  function fmtTime(t) {
    try { return new Date(typeof t === 'number' ? t * 1000 : t).toLocaleString(); }
    catch { return String(t); }
  }
  function formatUptime(sec) {
    if (sec === null || isNaN(sec)) return '--';
    const d = Math.floor(sec / (3600 * 24));
    const h = Math.floor(sec % (3600 * 24) / 3600);
    const m = Math.floor(sec % 3600 / 60);
    if (d > 0) return `${d}d ${h}h`;
    if (h > 0) return `${h}h ${m}m`;
    return `${m}m`;
  }
  function setPipImage(pip, url) {
    if (!url) return;
    pip.innerHTML = '';
    const img = document.createElement('img');
    img.decoding = 'async'; img.loading = 'lazy'; img.src = url;
    pip.appendChild(img);
  }
  function getImagePath(evt) {
    // ใช้โค้ดเดิมที่ยืดหยุ่น
    return evt?.image_path || evt?.image?.path || evt?.imagePath || null;
  }
  function getSide(evt) {
    const loc = (evt?.camera?.location || '').toLowerCase();
    if (loc === 'defence' || loc === 'offence') return loc;
    const camId = evt?.camera_id || evt?.cam_id || evt?.camera?.id;
    if (camId && CONFIG) {
      if (camId === CONFIG.defence.camera_id) return 'defence';
      if (camId === CONFIG.offence.camera_id) return 'offence';
    }
    return null;
  }

  // -------- Drone Tracking & Mapbox Helpers (NEW) --------
  function createDroneMarker(obj, map, side) {
    const color = (side === 'def') ? '#2bd6b6' : '#4aa3ff';

    // Custom Marker Element
    const el = document.createElement('div');
    el.className = 'drone-marker';
    el.style.backgroundColor = color;
    el.style.boxShadow = `0 0 10px ${color}`;

    // Custom Popup Content
    const popupContent = `
      <strong>ID:</strong> ${obj.obj_id}<br>
      <strong>Type:</strong> ${obj.type}<br>
      <strong>Objective:</strong> ${obj.objective || 'N/A'}<br>
      <strong>Size:</strong> ${obj.size || 'N/A'}<br>
      <strong>Lat/Lng:</strong> ${obj.lat.toFixed(5)}, ${obj.lng.toFixed(5)}
    `;

    const popup = new mapboxgl.Popup({ offset: 25, closeButton: false, closeOnClick: true })
      .setHTML(popupContent);

    const marker = new mapboxgl.Marker({ element: el })
      .setLngLat([obj.lng, obj.lat]) // Mapbox ใช้ [lng, lat]
      .setPopup(popup)
      .addTo(map);

    return marker;
  }

  function updateDroneMarkers(side, event) {
    const map = state[side].map;
    if (!map) return;

    const currentMarkers = state[side].markers;
    const detectedObjects = event.objects || [];
    const objectIds = new Set(detectedObjects.map(obj => obj.obj_id));

    // 1. อัปเดต/เพิ่ม Marker
    detectedObjects.forEach(obj => {
      if (obj.type !== 'drone') return;

      if (currentMarkers[obj.obj_id]) {
        // ย้าย Marker ที่มีอยู่
        currentMarkers[obj.obj_id].setLngLat([obj.lng, obj.lat]);
      } else {
        // สร้าง Marker ใหม่
        currentMarkers[obj.obj_id] = createDroneMarker(obj, map, side);
      }
    });

    // 2. ลบ Marker ที่หายไป
    for (const id in currentMarkers) {
      if (!objectIds.has(id)) {
        currentMarkers[id].remove();
        delete currentMarkers[id];
      }
    }

    // 3. เลื่อนแผนที่ไปที่โดรนตัวแรกที่ตรวจจับได้ (ถ้าต้องการให้แผนที่ตาม)
    // const firstDrone = detectedObjects.find(obj => obj.type === 'drone');
    // if (firstDrone) {
    //    map.panTo([firstDrone.lng, firstDrone.lat]);
    // }
  }


  // -------- Overview page --------
  async function initOverview() {
    const diag = document.getElementById('diag');
    const defSock = document.getElementById('def-socket');
    const offSock = document.getElementById('off-socket');

    const k = {
      defObj: document.getElementById('def-objects'),
      defTime: document.getElementById('def-time'),
      defLat: document.getElementById('def-lat'),
      defLng: document.getElementById('def-lng'),
      offObj: document.getElementById('off-objects'),
      offTime: document.getElementById('off-time'),
      offLat: document.getElementById('off-lat'),
      offLng: document.getElementById('off-lng'),
      hostTemp: document.getElementById('host-temp'),
      hostTime: document.getElementById('host-time'),
      hostUptime: document.getElementById('host-uptime'),
      hostStatus: document.getElementById('host-status'),
      hostSock: document.getElementById('host-sock'),
      hostApi: document.getElementById('host-api'),
      hostDefName: document.getElementById('host-def-name'),
      hostOffName: document.getElementById('host-off-name'),
    };

    const cfg = await loadConfig();
    k.hostSock.textContent = cfg.socket_url;
    k.hostApi.textContent = cfg.rest.image_base;

    // ดึงชื่อกล้องมาแสดง
    try {
      const [r1, r2] = await Promise.allSettled([
        axios.get(cfg.rest.defence.info, { timeout: 4000 }),
        axios.get(cfg.rest.offence.info, { timeout: 4000 })
      ]);
      if (r1.status === 'fulfilled' && r1.value.data?.data) {
        k.hostDefName.textContent = r1.value.data.data.name || 'Defence';
      } else { k.hostDefName.textContent = 'API Error'; }

      if (r2.status === 'fulfilled' && r2.value.data?.data) {
        k.hostOffName.textContent = r2.value.data.data.name || 'Offence';
      } else { k.hostOffName.textContent = 'API Error'; }

    } catch (e) {
      console.error("API Info Error:", e);
      k.hostDefName.textContent = k.hostOffName.textContent = 'API Info Error';
    }


    // host telemetry
    const pollHost = async () => {
      try {
        const { data } = await axios.get('/host/metrics', { timeout: 4000 });
        k.hostTemp.textContent = data.cpu_temp_c ? data.cpu_temp_c.toFixed(1) : 'N/A';
        k.hostTime.textContent = fmtTime(data.time);
        k.hostUptime.textContent = formatUptime(data.uptime_sec);
      } catch { }
    };
    pollHost(); setInterval(pollHost, 5000);

    // socket realtime
    socket = io(cfg.socket_url, { transports: ['websocket'] });
    socket.on('connect', () => {
      setBadge(defSock, true); setBadge(offSock, true);
      diag.textContent = `Diagnostics: Socket connected (${socket.id})`;
      if (cfg.defence.camera_id) socket.emit('subscribe_camera', { cam_id: cfg.defence.camera_id });
      if (cfg.offence.camera_id) socket.emit('subscribe_camera', { cam_id: cfg.offence.camera_id });
    });
    socket.on('disconnect', () => {
      setBadge(defSock, false); setBadge(offSock, false);
      diag.textContent = 'Diagnostics: Socket disconnected';
    });
    socket.on('object_detection', (ev) => {
      const side = getSide(ev);
      const n = Array.isArray(ev.objects) ? ev.objects.length : 0;
      // หาพิกัดโดรนตัวแรก
      const p = ev.objects?.find(o => o.type === 'drone') || {};
      const ts = ev.timestamp || Date.now();

      if (side === 'defence') {
        k.defObj.textContent = n; k.defTime.textContent = fmtTime(ts);
        k.defLat.textContent = p.lat ? p.lat.toFixed(5) : '--';
        k.defLng.textContent = p.lng ? p.lng.toFixed(5) : '--';
      } else if (side === 'offence') {
        k.offObj.textContent = n; k.offTime.textContent = fmtTime(ts);
        k.offLat.textContent = p.lat ? p.lat.toFixed(5) : '--';
        k.offLng.textContent = p.lng ? p.lng.toFixed(5) : '--';
      }
    });
  }

  // -------- Monitor page --------
  async function initMonitor() {
    const diag = document.getElementById('diag');
    const cfg = await loadConfig();

    // 1. Mapbox Initialization
    mapboxgl.accessToken = MAPBOX_TOKEN;
    const DEF_CENTER = [101.106, 14.259]; // [lng, lat]
    const OFF_CENTER = [100.5018, 13.7563];

    state.def.map = new mapboxgl.Map({
      container: 'def-map',
      style: 'mapbox://styles/mapbox/satellite-streets-v12',
      center: DEF_CENTER,
      zoom: 14,
    });
    state.off.map = new mapboxgl.Map({
      container: 'off-map',
      style: 'mapbox://styles/mapbox/dark-v11',
      center: OFF_CENTER,
      zoom: 12,
    });
    // เพิ่ม Navigation Control (Zoom/Rotate)
    state.def.map.addControl(new mapboxgl.NavigationControl(), 'bottom-right');
    state.off.map.addControl(new mapboxgl.NavigationControl(), 'bottom-right');

    const defPip = document.getElementById('def-pip');
    const offPip = document.getElementById('off-pip');
    const defSlot = document.getElementById('def-cam-slot');
    const offSlot = document.getElementById('off-cam-slot');
    const defPanel = document.getElementById('def-panel');
    const offPanel = document.getElementById('off-panel');
    const defBadge = document.getElementById('def-badge');
    const offBadge = document.getElementById('off-badge');

    // 2. Toggle Camera Expand Logic (อิสระต่อกัน)
    const toggleExpand = (key) => {
      const s = state[key];
      s.expanded = !s.expanded;
      const panel = key === 'def' ? defPanel : offPanel;
      const slot = key === 'def' ? defSlot : offSlot;
      const pip = key === 'def' ? defPip : offPip;
      const mapWrap = panel.querySelector('.map-wrap');

      panel.classList.toggle('camera-expanded', s.expanded);

      if (s.expanded) {
        slot.style.display = 'block';
        slot.appendChild(pip);
        s.map.resize(); // **สำคัญ**: ต้องเรียก resize ให้ Mapbox ปรับขนาด
      }
      else {
        slot.style.display = 'none';
        mapWrap.appendChild(pip);
        s.map.resize(); // **สำคัญ**: ต้องเรียก resize ให้ Mapbox ปรับขนาด
      }
    };
    defPip.onclick = () => toggleExpand('def');
    offPip.onclick = () => toggleExpand('off');

    // 3. Preload recent (และ Tracking โดรนเริ่มต้น)
    try {
      const [r1, r2] = await Promise.allSettled([
        axios.get(cfg.rest.defence.recent, { params: { limit: 1 }, timeout: 8000 }),
        axios.get(cfg.rest.offence.recent, { params: { limit: 1 }, timeout: 8000 })
      ]);

      let defStatus = 'Fail'; let offStatus = 'Fail';

      // ** DEFENCE **
      if (r1.status === 'fulfilled' && r1.value.data?.data?.[0]) {
        defStatus = 'OK';
        const d0 = r1.value.data.data[0];
        if (getImagePath(d0)) { setPipImage(defPip, cfg.rest.image_base + getImagePath(d0)); }

        updateDroneMarkers('def', d0);
        // เลื่อนแผนที่ไปที่โดรนตัวแรกที่ตรวจจับได้
        const d_drone = d0.objects?.find(o => o.type === 'drone');
        if (d_drone) state.def.map.setCenter([d_drone.lng, d_drone.lat]);

      } else if (r1.status === 'rejected') {
        defStatus = 'Proxy Error';
      }

      // ** OFFENCE **
      if (r2.status === 'fulfilled' && r2.value.data?.data?.[0]) {
        offStatus = 'OK';
        const o0 = r2.value.data.data[0];
        if (getImagePath(o0)) { setPipImage(offPip, cfg.rest.image_base + getImagePath(o0)); }

        updateDroneMarkers('off', o0);
        // เลื่อนแผนที่ไปที่โดรนตัวแรกที่ตรวจจับได้
        const o_drone = o0.objects?.find(o => o.type === 'drone');
        if (o_drone) state.off.map.setCenter([o_drone.lng, o_drone.lat]);
      } else if (r2.status === 'rejected') {
        offStatus = 'Proxy Error';
      }

      diag.textContent = `Socket: … | Defence API: ${defStatus} | Offence API: ${offStatus}`;
    } catch (e) {
      console.error("API Preload Error:", e);
      diag.textContent = 'Socket: … | API Preload Error';
    }

    // 4. Socket Realtime Tracking
    socket = io(cfg.socket_url, { transports: ['websocket'] });
    socket.on('connect', () => {
      setBadge(defBadge, true); setBadge(offBadge, true);
      diag.textContent = `Socket: connected (${socket.id}) | Defence API: OK | Offence API: OK`;
      if (cfg.defence.camera_id) socket.emit('subscribe_camera', { cam_id: cfg.defence.camera_id });
      if (cfg.offence.camera_id) socket.emit('subscribe_camera', { cam_id: cfg.offence.camera_id });
    });
    socket.on('disconnect', () => {
      setBadge(defBadge, false); setBadge(offBadge, false);
      diag.textContent = 'Socket: disconnected';
    });
    socket.on('object_detection', (ev) => {
      const side = getSide(ev);
      const ip = getImagePath(ev);
      const full = ip ? (CONFIG.rest.image_base + ip) : null;

      if (side === 'defence') {
        if (full) setPipImage(defPip, full);
        updateDroneMarkers('def', ev);
      } else if (side === 'offence') {
        if (full) setPipImage(offPip, full);
        updateDroneMarkers('off', ev);
      }
    });
  }

  return { initOverview, initMonitor };
})();