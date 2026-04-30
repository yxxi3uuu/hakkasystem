// ── 登入狀態管理 ──

function getUser() {
  const id = localStorage.getItem('user_id');
  if (!id) return null;
  return {
    id: parseInt(id),
    name: localStorage.getItem('user_name') || '',
    email: localStorage.getItem('user_email') || '',
  };
}

function requireLogin() {
  if (!getUser()) {
    window.location.href = '/login';
    return null;
  }
  return getUser();
}

async function logout() {
  try {
    await fetch('/api/auth/logout', { method: 'POST' });
  } catch (e) {}
  localStorage.removeItem('user_id');
  localStorage.removeItem('user_name');
  localStorage.removeItem('user_email');
  window.location.href = '/login';
}
