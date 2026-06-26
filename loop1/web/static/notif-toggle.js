/**
 * Shared patient desktop-notification toggle.
 * Include after auth.js. Auto-injects button into .nav before #nav-auth.
 */
(function () {
  // ── CSS ──────────────────────────────────────────────────────────────────
  const style = document.createElement('style');
  style.textContent = `
    .notif-toggle-btn {
      display: flex; align-items: center; gap: 7px;
      background: var(--surface-3, #1a2135); border: 1px solid rgba(255,255,255,.09);
      border-radius: 99px; padding: 5px 12px 5px 9px;
      cursor: pointer; color: var(--text-muted, #8896b3);
      font-size: 12px; font-weight: 600; font-family: inherit;
      transition: all .15s; white-space: nowrap; flex-shrink: 0;
    }
    .notif-toggle-btn:hover { border-color: rgba(0,180,216,.4); color: var(--text, #e2e8f0); }
    .notif-toggle-btn.enabled {
      background: rgba(0,180,216,.1); border-color: rgba(0,180,216,.35);
      color: var(--primary, #00b4d8);
    }
    .notif-toggle-btn.denied {
      background: rgba(239,68,68,.07); border-color: rgba(239,68,68,.25);
      color: #ef4444; cursor: not-allowed;
    }
    .notif-bell-dot {
      width: 6px; height: 6px; border-radius: 50%;
      background: var(--primary, #00b4d8); flex-shrink: 0;
      box-shadow: 0 0 6px rgba(0,180,216,.7);
      animation: _nb-pulse 1.8s ease infinite;
    }
    @keyframes _nb-pulse {
      0%,100% { opacity:1; transform:scale(1); }
      50%      { opacity:.5; transform:scale(.7); }
    }
  `;
  document.head.appendChild(style);

  // ── State ─────────────────────────────────────────────────────────────────
  const PREF_KEY = 'push_notif_enabled';
  const SEEN_KEY = 'push_notif_seen_' + new Date().toISOString().slice(0, 10);

  // Expose globally so page scripts can call _firePushNotif etc.
  window._patNotifSeen = new Set(
    JSON.parse(localStorage.getItem(SEEN_KEY) || '[]')
  );

  window._savePatNotifSeen = function () {
    localStorage.setItem(SEEN_KEY, JSON.stringify([...window._patNotifSeen]));
  };

  window._isNotifEnabled = function () {
    return localStorage.getItem(PREF_KEY) === '1' &&
           typeof Notification !== 'undefined' &&
           Notification.permission === 'granted';
  };

  window._firePushNotif = function (id, title, body, opts) {
    opts = opts || {};
    if (!window._isNotifEnabled()) return;
    if (window._patNotifSeen.has(id)) return;
    window._patNotifSeen.add(id);
    window._savePatNotifSeen();
    const n = new Notification(title, {
      body, icon: '/favicon.ico', badge: '/favicon.ico', tag: id,
      requireInteraction: opts.requireInteraction || false,
    });
    n.onclick = function () {
      window.focus();
      if (opts.link) window.location.href = opts.link;
      n.close();
    };
  };

  window._updateNotifBtn = function () {
    const btn  = document.getElementById('notif-toggle-btn');
    const lbl  = document.getElementById('notif-toggle-label');
    const dot  = document.getElementById('notif-bell-dot');
    const icon = document.getElementById('notif-bell-icon');
    if (!btn) return;
    if (typeof Notification === 'undefined' || Notification.permission === 'denied') {
      btn.className = 'notif-toggle-btn denied';
      lbl.textContent = 'Blocked';
      dot.style.display = 'none';
      icon.setAttribute('stroke', '#ef4444');
      btn.title = 'Notifications blocked — allow in browser site settings';
      return;
    }
    const on = window._isNotifEnabled();
    btn.className = 'notif-toggle-btn' + (on ? ' enabled' : '');
    lbl.textContent = on ? 'Notifications on' : 'Notifications';
    dot.style.display = on ? '' : 'none';
    icon.setAttribute('stroke', on ? 'var(--primary, #00b4d8)' : 'currentColor');
    btn.title = on ? 'Click to disable desktop notifications'
                   : 'Click to enable desktop notifications';
  };

  window.toggleBrowserNotif = async function () {
    if (typeof Notification === 'undefined') return;
    if (Notification.permission === 'denied') {
      alert('Notifications are blocked. Allow them in your browser site settings, then reload.');
      return;
    }
    if (window._isNotifEnabled()) {
      localStorage.setItem(PREF_KEY, '0');
      window._updateNotifBtn();
      return;
    }
    if (Notification.permission !== 'granted') {
      const result = await Notification.requestPermission();
      if (result !== 'granted') { window._updateNotifBtn(); return; }
    }
    localStorage.setItem(PREF_KEY, '1');
    window._updateNotifBtn();
    new Notification('DiffDx', {
      body: "You'll get desktop alerts for appointments, messages, and test results.",
      icon: '/favicon.ico', tag: 'welcome-patient',
    });
  };

  // ── Inject button into nav ────────────────────────────────────────────────
  function injectButton() {
    if (document.getElementById('notif-toggle-btn')) return; // already present
    const navAuth = document.getElementById('nav-auth');
    const spacer  = document.querySelector('.nav-spacer');
    const anchor  = navAuth || spacer;
    if (!anchor) return;

    const btn = document.createElement('button');
    btn.id = 'notif-toggle-btn';
    btn.className = 'notif-toggle-btn';
    btn.onclick = window.toggleBrowserNotif;
    btn.style.marginRight = '8px';
    btn.innerHTML = `
      <svg id="notif-bell-icon" width="14" height="14" viewBox="0 0 24 24"
           fill="none" stroke="currentColor" stroke-width="2.2">
        <path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9"/>
        <path d="M13.73 21a2 2 0 0 1-3.46 0"/>
      </svg>
      <span id="notif-toggle-label">Notifications</span>
      <span id="notif-bell-dot" class="notif-bell-dot" style="display:none;"></span>
    `;

    // Insert before nav-auth (or before spacer's next sibling)
    if (navAuth) {
      navAuth.parentNode.insertBefore(btn, navAuth);
    } else {
      spacer.parentNode.insertBefore(btn, spacer.nextSibling);
    }

    window._updateNotifBtn();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', injectButton);
  } else {
    injectButton();
  }
})();
