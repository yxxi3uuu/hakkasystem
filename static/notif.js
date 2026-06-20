// notif.js — 站內通知共用模組
// 在每個頁面 <script src="/static/notif.js"></script> 後呼叫 initNotif()

function initNotif() {
  const user = getUser();
  if (!user) return;

  // 在 top-bar 插入鈴鐺按鈕
  const topBar = document.querySelector('.top-bar');
  if (!topBar) return;

  const btn = document.createElement('button');
  btn.className = 'notif-btn';
  btn.id = 'notifBtn';
  btn.title = '通知';
  btn.innerHTML = `🔔<span class="notif-badge" id="notifBadge"></span>`;
  btn.onclick = openNotifPanel;
  topBar.appendChild(btn);

  // 建立通知彈窗
  if (!document.getElementById('notifOverlay')) {
    const overlay = document.createElement('div');
    overlay.className = 'notif-overlay';
    overlay.id = 'notifOverlay';
    overlay.onclick = (e) => { if (e.target === overlay) closeNotifPanel(); };
    overlay.innerHTML = `
      <div class="notif-panel">
        <div class="notif-panel-head">
          <span class="notif-panel-title">🔔 通知</span>
          <button class="notif-read-all" onclick="markAllRead()">全部標為已讀</button>
        </div>
        <div id="notifList"><div class="notif-empty">載入中...</div></div>
      </div>`;
    document.body.appendChild(overlay);
  }

  refreshUnreadCount();
  // 每 30 秒自動更新未讀數
  setInterval(refreshUnreadCount, 30000);
}

async function refreshUnreadCount() {
  const user = getUser();
  if (!user) return;
  try {
    const data = await fetch(`/api/notifications/unread-count?user_id=${user.id}`).then(r => r.json());
    const badge = document.getElementById('notifBadge');
    if (!badge) return;
    if (data.count > 0) {
      badge.textContent = data.count > 99 ? '99+' : data.count;
      badge.classList.add('show');
    } else {
      badge.classList.remove('show');
    }
  } catch(e) {}
}

async function openNotifPanel() {
  const user = getUser();
  if (!user) return;
  document.getElementById('notifOverlay').classList.add('open');
  document.body.style.overflow = 'hidden';
  await loadNotifs();
}

function closeNotifPanel() {
  document.getElementById('notifOverlay').classList.remove('open');
  document.body.style.overflow = '';
}

async function loadNotifs() {
  const user = getUser();
  const list = document.getElementById('notifList');
  try {
    const notifs = await fetch(`/api/notifications?user_id=${user.id}`).then(r => r.json());
    if (!notifs.length) {
      list.innerHTML = '<div class="notif-empty">目前沒有通知</div>';
      return;
    }
    list.innerHTML = notifs.map(n => `
      <div class="notif-item ${n.is_read ? '' : 'unread'}" onclick="readNotif(${n.id})">
        <div class="notif-dot ${n.is_read ? 'read' : ''}"></div>
        <div style="flex:1">
          <div class="notif-item-title">${n.title}</div>
          ${n.body ? `<div class="notif-item-body">${n.body}</div>` : ''}
          <div class="notif-item-meta">${n.sender_name ? `來自 ${n.sender_name}　` : ''}${n.created_at}</div>
        </div>
        <button onclick="event.stopPropagation();deleteNotif(${n.id},this)" style="background:none;border:none;color:#ccc;cursor:pointer;font-size:16px;padding:0 0 0 8px;">✕</button>
      </div>`).join('');
  } catch(e) {
    list.innerHTML = '<div class="notif-empty">載入失敗</div>';
  }
}

async function readNotif(id) {
  const user = getUser();
  await fetch(`/api/notifications/${id}/read?user_id=${user.id}`, { method: 'POST' });
  const item = document.querySelector(`[onclick="readNotif(${id})"]`);
  if (item) {
    item.classList.remove('unread');
    const dot = item.querySelector('.notif-dot');
    if (dot) dot.classList.add('read');
  }
  refreshUnreadCount();
}

async function markAllRead() {
  const user = getUser();
  await fetch(`/api/notifications/read-all?user_id=${user.id}`, { method: 'POST' });
  await loadNotifs();
  refreshUnreadCount();
}

async function deleteNotif(id, btn) {
  const user = getUser();
  await fetch(`/api/notifications/${id}?user_id=${user.id}`, { method: 'DELETE' });
  btn.closest('.notif-item').remove();
  refreshUnreadCount();
  const list = document.getElementById('notifList');
  if (!list.querySelector('.notif-item')) {
    list.innerHTML = '<div class="notif-empty">目前沒有通知</div>';
  }
}
