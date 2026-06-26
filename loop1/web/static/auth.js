/**
 * Shared auth helpers — included by every page.
 * Reads/writes localStorage keys: authToken, authUser, authExpiry.
 */

const _AUTH_TOKEN_KEY  = 'authToken';
const _AUTH_USER_KEY   = 'authUser';
const _AUTH_EXPIRY_KEY = 'authExpiry';
const _SESSION_TTL_MS  = 8 * 60 * 60 * 1000; // 8 hours

function getAuthToken() { return localStorage.getItem(_AUTH_TOKEN_KEY); }
function getAuthUser() {
  const raw = localStorage.getItem(_AUTH_USER_KEY);
  try { return raw ? JSON.parse(raw) : null; } catch { return null; }
}
function setAuth(token, user) {
  localStorage.setItem(_AUTH_TOKEN_KEY, token);
  localStorage.setItem(_AUTH_USER_KEY, JSON.stringify(user));
  localStorage.setItem(_AUTH_EXPIRY_KEY, String(Date.now() + _SESSION_TTL_MS));
}
function clearAuth() {
  localStorage.removeItem(_AUTH_TOKEN_KEY);
  localStorage.removeItem(_AUTH_USER_KEY);
  localStorage.removeItem(_AUTH_EXPIRY_KEY);
}
function authHeaders() {
  const t = getAuthToken();
  return t ? { 'Authorization': `Bearer ${t}` } : {};
}
function logout() {
  clearAuth();
  window.location.href = '/';
}

/** Check expiry and show session-expired toast if needed. */
function _checkSessionExpiry() {
  const token = getAuthToken();
  if (!token) return;
  const expiry = parseInt(localStorage.getItem(_AUTH_EXPIRY_KEY) || '0', 10);
  if (expiry && Date.now() > expiry) {
    clearAuth();
    _showSessionExpiredToast();
    return;
  }
  // Re-check every 60 seconds
  setTimeout(_checkSessionExpiry, 60_000);
}

function _showSessionExpiredToast() {
  let toast = document.getElementById('_session-expired-toast');
  if (!toast) {
    toast = document.createElement('div');
    toast.id = '_session-expired-toast';
    toast.style.cssText = `
      position:fixed;bottom:24px;left:50%;transform:translateX(-50%);
      background:#1e2a38;border:1px solid rgba(239,68,68,.4);border-radius:10px;
      padding:14px 22px;display:flex;align-items:center;gap:12px;
      box-shadow:0 8px 32px rgba(0,0,0,.5);z-index:99999;
      font-family:Inter,sans-serif;font-size:13px;color:#f1f5f9;
      animation:_toastIn .25s ease;
    `;
    const style = document.createElement('style');
    style.textContent = `@keyframes _toastIn{from{opacity:0;transform:translateX(-50%) translateY(12px)}to{opacity:1;transform:translateX(-50%) translateY(0)}}`;
    document.head.appendChild(style);
    toast.innerHTML = `
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#ef4444" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
      <span>Your session has expired.</span>
      <a href="/login.html" style="color:#00b4d8;font-weight:600;text-decoration:none;margin-left:4px;">Sign in again</a>
    `;
    document.body.appendChild(toast);
  }
  toast.style.display = 'flex';
}

/**
 * Intercept fetch calls — if a 401 is returned after the user was logged in,
 * show the session-expired toast instead of silently failing.
 */
const _origFetch = window.fetch;
window.fetch = async function(...args) {
  const res = await _origFetch(...args);
  if (res.status === 401 && getAuthToken()) {
    clearAuth();
    _showSessionExpiredToast();
    return res;
  }
  return res;
};

function _escHtml(s) {
  return (s || '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

/** Inject auth state into any element with id="nav-auth" */
function initAuthNav() {
  _checkSessionExpiry();
  const el = document.getElementById('nav-auth');
  if (!el) return;
  const user = getAuthUser();
  if (user) {
    if (user.role === 'doctor') {
      el.innerHTML = `
        <a href="/doctor-portal.html" style="position:relative;display:flex;align-items:center;justify-content:center;width:32px;height:32px;border-radius:8px;border:1px solid rgba(255,255,255,.09);background:rgba(255,255,255,.04);color:var(--text-muted);transition:all .15s;" title="Pending Refill Requests" id="nav-refill-btn">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="1 4 1 10 7 10"/><path d="M3.51 15a9 9 0 102.13-9.36L1 10"/></svg>
          <span id="nav-refill-badge" style="display:none;position:absolute;top:-4px;right:-4px;background:#f59e0b;color:#000;font-size:9px;font-weight:800;border-radius:99px;padding:1px 5px;line-height:1.6;min-width:16px;text-align:center;"></span>
        </a>
        <div class="nav-user-menu" id="nav-user-menu">
          <button class="nav-user-btn" onclick="_toggleUserMenu(event)" aria-expanded="false">
            <span class="nav-user-avatar">${_escHtml(user.name.charAt(0).toUpperCase())}</span>
            <span class="nav-user-name">${_escHtml(user.name)}</span>
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="6 9 12 15 18 9"/></svg>
          </button>
          <div class="nav-user-dropdown" id="nav-user-dropdown">
            <div class="nav-user-dropdown-header">
              <div style="font-weight:600;font-size:13px;">${_escHtml(user.name)}</div>
              <div style="font-size:11px;color:var(--text-muted);">${_escHtml(user.email || '')}</div>
            </div>
            <a class="nav-user-dropdown-item" href="/doctor-profile.html">My Profile</a>
            <a class="nav-user-dropdown-item" href="/doctor-analytics.html">Analytics</a>
            <a class="nav-user-dropdown-item" href="/messages.html">Messages <span id="nav-msg-badge" style="display:none;margin-left:auto;background:#ef4444;color:#fff;font-size:9px;font-weight:800;border-radius:99px;padding:1px 6px;"></span></a>
            <button class="nav-user-dropdown-item nav-user-dropdown-item--danger" onclick="logout()">Sign Out</button>
          </div>
        </div>
      `;
      _loadDoctorRefillBadge();
    } else {
      el.innerHTML = `
        <a href="/my-sessions.html" class="btn btn-ghost btn-sm" style="font-size:12px;padding:5px 10px;" id="nav-sessions-link">
          My Sessions
        </a>
        <a href="/notifications.html" style="position:relative;display:flex;align-items:center;justify-content:center;width:32px;height:32px;border-radius:8px;border:1px solid rgba(255,255,255,.09);background:rgba(255,255,255,.04);color:var(--text-muted);transition:all .15s;" title="Notifications" id="nav-notif-btn">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M18 8A6 6 0 006 8c0 7-3 9-3 9h18s-3-2-3-9"/><path d="M13.73 21a2 2 0 01-3.46 0"/></svg>
          <span id="nav-notif-badge" style="display:none;position:absolute;top:-4px;right:-4px;background:#00b4d8;color:#000;font-size:9px;font-weight:800;border-radius:99px;padding:1px 5px;line-height:1.6;min-width:16px;text-align:center;"></span>
        </a>
        <div class="nav-user-menu" id="nav-user-menu">
          <button class="nav-user-btn" onclick="_toggleUserMenu(event)" aria-expanded="false">
            <span class="nav-user-avatar">${_escHtml(user.name.charAt(0).toUpperCase())}</span>
            <span class="nav-user-name">${_escHtml(user.name)}</span>
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="6 9 12 15 18 9"/></svg>
          </button>
          <div class="nav-user-dropdown" id="nav-user-dropdown">
            <div class="nav-user-dropdown-header">
              <div style="font-weight:600;font-size:13px;">${_escHtml(user.name)}</div>
              <div style="font-size:11px;color:var(--text-muted);">${_escHtml(user.email || '')}</div>
            </div>
            <a class="nav-user-dropdown-item" href="/my-profile.html">My Profile</a>
            <a class="nav-user-dropdown-item" href="/health-history.html">Health History</a>
            <a class="nav-user-dropdown-item" href="/messages.html">Messages <span id="nav-msg-badge" style="display:none;margin-left:auto;background:#ef4444;color:#fff;font-size:9px;font-weight:800;border-radius:99px;padding:1px 6px;"></span></a>
            <button class="nav-user-dropdown-item nav-user-dropdown-item--danger" onclick="logout()">Sign Out</button>
          </div>
        </div>
      `;
      _loadTestNotifBadge();
    }
    _loadMsgBadge();
  } else {
    el.innerHTML = `
      <a href="/login.html" class="btn btn-ghost btn-sm" style="font-size:12px;padding:5px 10px;">Sign In</a>
    `;
  }
}

function _toggleUserMenu(e) {
  e.stopPropagation();
  const dropdown = document.getElementById('nav-user-dropdown');
  const btn = e.currentTarget;
  const open = dropdown.classList.toggle('open');
  btn.setAttribute('aria-expanded', open);
  if (open) {
    const close = (ev) => {
      if (!document.getElementById('nav-user-menu')?.contains(ev.target)) {
        dropdown.classList.remove('open');
        btn.setAttribute('aria-expanded', 'false');
        document.removeEventListener('click', close);
      }
    };
    document.addEventListener('click', close);
  }
}

function _seenTestIds() {
  try { return new Set(JSON.parse(localStorage.getItem('seen_test_notifs') || '[]')); } catch { return new Set(); }
}
function _markTestIdsSeen(ids) {
  try {
    const seen = _seenTestIds();
    ids.forEach(id => seen.add(id));
    localStorage.setItem('seen_test_notifs', JSON.stringify([...seen]));
  } catch {}
}

async function _loadTestNotifBadge() {
  try {
    const [notifRes, apptRes, msgRes] = await Promise.all([
      _origFetch('/api/patient/test-notifications', { headers: { ...authHeaders() } }),
      _origFetch('/api/appointments', { headers: { ...authHeaders() } }),
      _origFetch('/api/messages/unread-count', { headers: { ...authHeaders() } }),
    ]);

    // Test orders badge (My Sessions button)
    if (notifRes.ok) {
      const data = await notifRes.json();
      const seen = _seenTestIds();
      const unseen = (data.appointments || []).filter(a => !seen.has(a.appt_id));
      const count = unseen.reduce((s, a) => s + (a.test_count || 0), 0);
      if (count > 0) {
        const badge = document.getElementById('test-notif-badge');
        if (badge) { badge.textContent = count; badge.style.display = 'inline'; }
      }
    }

    // Bell badge = unread notifications across all types
    const seenNotifIds = (() => { try { return new Set(JSON.parse(localStorage.getItem('notif_seen_ids') || '[]')); } catch { return new Set(); } })();
    let bellCount = 0;
    if (apptRes.ok) {
      const apptData = await apptRes.json();
      for (const a of (apptData.appointments || [])) {
        if (a.referral && !seenNotifIds.has(`ref-${a.id}`)) bellCount++;
        if (a.test_orders?.length && !seenNotifIds.has(`tests-${a.id}`)) bellCount++;
        if (a.prescriptions?.length && !seenNotifIds.has(`rx-${a.id}`)) bellCount++;
        if (a.refill_request?.status === 'fulfilled' && !seenNotifIds.has(`refill-${a.id}`)) bellCount++;
      }
    }
    if (msgRes.ok) { const d = await msgRes.json(); if (d.count > 0 && !seenNotifIds.has('unread-messages')) bellCount++; }
    if (bellCount > 0) {
      const badge = document.getElementById('nav-notif-badge');
      if (badge) { badge.textContent = bellCount; badge.style.display = 'inline'; }
    }
  } catch { /* ignore */ }
}

async function _loadDoctorRefillBadge() {
  try {
    const res = await _origFetch('/api/doctor/pending-refills', { headers: { ...authHeaders() } });
    if (!res.ok) return;
    const data = await res.json();
    if (data.pending > 0) {
      const badge = document.getElementById('nav-refill-badge');
      if (badge) { badge.textContent = data.pending; badge.style.display = 'inline'; }
    }
  } catch { /* ignore */ }
}

async function _loadMsgBadge() {
  try {
    const res = await _origFetch('/api/messages/unread-count', { headers: { ...authHeaders() } });
    if (!res.ok) return;
    const data = await res.json();
    if (data.count > 0) {
      const badge = document.getElementById('nav-msg-badge');
      if (badge) { badge.textContent = data.count; badge.style.display = 'inline'; }
    }
  } catch { /* ignore */ }
}

document.addEventListener('DOMContentLoaded', initAuthNav);
