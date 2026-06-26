/**
 * Shared patient tab navigation.
 * Include after auth.js on every patient-facing page.
 * Auto-injects a tab bar below the nav and standardises page headers.
 */
(function () {
  const TABS = [
    {
      id: 'home',
      label: 'Home',
      href: '/my-sessions.html',
      match: ['/my-sessions.html', '/index-patient'],
      icon: `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><polyline points="9 22 9 12 15 12 15 22"/></svg>`,
    },
    {
      id: 'book',
      label: 'Book',
      href: '/book-slot.html',
      match: ['/book-slot.html'],
      icon: `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="4" width="18" height="18" rx="2"/><line x1="16" y1="2" x2="16" y2="6"/><line x1="8" y1="2" x2="8" y2="6"/><line x1="3" y1="10" x2="21" y2="10"/></svg>`,
    },
    {
      id: 'messages',
      label: 'Messages',
      href: '/messages.html',
      match: ['/messages.html'],
      icon: `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>`,
    },
    {
      id: 'notifications',
      label: 'Notifications',
      href: '/notifications.html',
      match: ['/notifications.html'],
      icon: `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9"/><path d="M13.73 21a2 2 0 0 1-3.46 0"/></svg>`,
    },
    {
      id: 'history',
      label: 'History',
      href: '/health-history.html',
      match: ['/health-history.html'],
      icon: `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><polyline points="12 7 12 12 15 15"/></svg>`,
    },
    {
      id: 'profile',
      label: 'Profile',
      href: '/my-profile.html',
      match: ['/my-profile.html'],
      icon: `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>`,
    },
  ];

  // ── CSS ────────────────────────────────────────────────────────────────
  const style = document.createElement('style');
  style.textContent = `
    .patient-tabbar {
      position: sticky; top: 56px; z-index: 90;
      background: rgba(7,9,15,.92);
      backdrop-filter: blur(20px) saturate(1.6);
      -webkit-backdrop-filter: blur(20px) saturate(1.6);
      border-bottom: 1px solid rgba(255,255,255,.06);
      display: flex; align-items: center;
      padding: 0 24px; gap: 2px;
      height: 44px;
    }
    .patient-tab {
      display: inline-flex; align-items: center; gap: 7px;
      padding: 6px 14px; border-radius: 8px;
      font-size: 12.5px; font-weight: 600; color: var(--text-muted, #64748b);
      cursor: pointer; border: none; background: none;
      font-family: inherit; text-decoration: none;
      transition: color .15s, background .15s;
      position: relative; white-space: nowrap;
    }
    .patient-tab:hover { color: var(--text, #e2e8f0); background: rgba(255,255,255,.04); }
    .patient-tab.active { color: var(--primary, #00b4d8); }
    .patient-tab.active::after {
      content: ''; position: absolute; bottom: -11px; left: 14px; right: 14px;
      height: 2px; border-radius: 2px 2px 0 0;
      background: var(--primary, #00b4d8);
      box-shadow: 0 0 8px rgba(0,180,216,.5);
    }
    .patient-tab-badge {
      display: inline-flex; align-items: center; justify-content: center;
      min-width: 16px; height: 16px; border-radius: 99px;
      background: var(--primary, #00b4d8); color: #07090f;
      font-size: 9px; font-weight: 800; padding: 0 4px;
    }

    /* Standardise page header across all patient pages */
    .patient-page-header {
      display: flex; align-items: flex-start; justify-content: space-between;
      gap: 16px; margin-bottom: 24px; flex-wrap: wrap;
    }
    .patient-page-title {
      font-size: 22px; font-weight: 800; letter-spacing: -.03em;
      color: var(--text, #e2e8f0); line-height: 1.2; margin-bottom: 3px;
    }
    .patient-page-sub {
      font-size: 13px; color: var(--text-muted, #64748b);
    }
    @media (max-width: 600px) {
      .patient-tabbar { padding: 0 12px; gap: 0; overflow-x: auto; }
      .patient-tab { font-size: 11px; padding: 6px 10px; gap: 5px; }
      .patient-tab svg { display: none; }
    }
  `;
  document.head.appendChild(style);

  // ── Inject tab bar ─────────────────────────────────────────────────────
  function injectTabBar() {
    if (document.getElementById('patient-tabbar')) return;
    const nav = document.querySelector('nav.nav');
    if (!nav) return;

    const path = location.pathname;
    const bar  = document.createElement('nav');
    bar.id = 'patient-tabbar';
    bar.className = 'patient-tabbar';

    bar.innerHTML = TABS.map(t => {
      const isActive = t.match.some(m => path === m || path.startsWith(m));
      return `<a class="patient-tab${isActive ? ' active' : ''}" href="${t.href}" id="ptab-${t.id}">
        ${t.icon}${t.label}
      </a>`;
    }).join('');

    nav.insertAdjacentElement('afterend', bar);

    // Load unread badge for messages
    _loadPatientTabBadges();
  }

  async function _loadPatientTabBadges() {
    try {
      if (typeof authHeaders !== 'function') return;
      const r = await fetch('/api/messages/unread-count', { headers: authHeaders() });
      if (!r.ok) return;
      const { count } = await r.json();
      if (count > 0) {
        const tab = document.getElementById('ptab-messages');
        if (tab) {
          const badge = document.createElement('span');
          badge.className = 'patient-tab-badge';
          badge.textContent = count > 9 ? '9+' : count;
          tab.appendChild(badge);
        }
      }
    } catch (_) {}
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', injectTabBar);
  } else {
    injectTabBar();
  }
})();
