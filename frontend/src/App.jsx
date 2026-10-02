import React, { useCallback, useEffect, useMemo, useState } from "react";
import { api, getToken, jsonBody, setToken } from "./api";

const CATEGORIES = ["AI/ML", "Web/Software", "Cybersecurity", "Robotics/IoT", "Sports", "Cultural", "Entrepreneurship", "Higher Studies/Career"];
const DEPARTMENTS = ["Computer Science", "Information Technology", "Electronics", "Mechanical", "Civil"];
const dateText = (value) => value ? new Intl.DateTimeFormat("en-IN", { dateStyle: "medium", timeStyle: "short" }).format(new Date(value)) : "Date to be announced";
const moneyText = (fee) => Number(fee) === 0 ? "Free" : `₹${Number(fee).toLocaleString("en-IN")}`;
const initials = (name = "") => name.split(/\s+/).slice(0, 2).map((part) => part[0]).join("").toUpperCase();

function EventCard({ event, reason, onOpen, onRegister, onLike }) {
  const registrationClosed = new Date(event.registration_deadline) < new Date() || new Date(event.start_time) < new Date();
  return <article className="event-card" onClick={onOpen}>
    <div className="event-card-top"><span className="category-pill">{event.category}</span><span className="event-fee">{moneyText(event.fee)}</span></div>
    <h3>{event.title}</h3>
    <p className="event-description">{event.description}</p>
    <div className="event-meta"><span>◷ {dateText(event.start_time)}</span><span>⌖ {event.venue}</span></div>
    {reason && <div className="reason-line"><span className="sparkle">✦</span> {reason}</div>}
    <div className="card-actions" onClick={(eventClick) => eventClick.stopPropagation()}>
      <button className="button button-primary button-small" disabled={registrationClosed} onClick={onRegister}>{registrationClosed ? "Closed" : "Register"}</button>
      <button className="button button-quiet button-small" aria-label={`Like ${event.title}`} onClick={onLike}>♡ Like</button>
    </div>
  </article>;
}

function AuthScreen({ onLogin, onRegister, busy, error }) {
  const [mode, setMode] = useState("login");
  const [form, setForm] = useState({ name: "", email: "", password: "", department: DEPARTMENTS[0], year: 1 });
  const update = (key) => (event) => setForm((current) => ({ ...current, [key]: event.target.value }));
  const submit = (event) => { event.preventDefault(); mode === "login" ? onLogin(form) : onRegister(form); };
  return <main className="auth-screen">
    <div className="auth-art"><div className="brand brand-light"><span className="brand-mark">e</span><span>evently</span></div>
      <div className="auth-copy"><p className="eyebrow">YOUR CAMPUS, IN MOTION</p><h1>Find your<br /><em>next thing.</em></h1>
        <p>Good events bring people together. Find the workshops, ideas and people worth showing up for.</p>
        <div className="orbit orbit-one"/><div className="orbit orbit-two"/><div className="orbit-dot">✦</div>
      </div>
      <span className="auth-art-footer">A little more campus, a lot more connection.</span>
    </div>
    <div className="auth-panel"><form className="auth-form" onSubmit={submit}>
      <p className="eyebrow">{mode === "login" ? "WELCOME BACK" : "GET STARTED"}</p>
      <h2>{mode === "login" ? "Sign in to evently" : "Create your account"}</h2>
      <p className="muted">{mode === "login" ? "Your campus calendar is waiting." : "Join your campus community."}</p>
      {error && <div className="alert alert-error">{error}</div>}
      {mode === "register" && <label>Full name<input required value={form.name} onChange={update("name")} placeholder="Aniket Sharma" /></label>}
      <label>College email<input required type="email" value={form.email} onChange={update("email")} placeholder="you@college.edu" /></label>
      <label>Password<input required type="password" minLength="8" value={form.password} onChange={update("password")} placeholder="At least 8 characters" /></label>
      {mode === "register" && <div className="form-row">
        <label>Department<select value={form.department} onChange={update("department")}>{DEPARTMENTS.map((item) => <option key={item}>{item}</option>)}</select></label>
        <label>Year<select value={form.year} onChange={update("year")}>{[1, 2, 3, 4].map((item) => <option key={item} value={item}>Year {item}</option>)}</select></label>
      </div>}
      <button className="button button-primary button-wide" disabled={busy}>{busy ? "Please wait…" : mode === "login" ? "Sign in" : "Create account"}<span>→</span></button>
      <p className="auth-switch">{mode === "login" ? "New to evently?" : "Already have an account?"} <button type="button" onClick={() => setMode(mode === "login" ? "register" : "login")}>{mode === "login" ? "Create account" : "Sign in"}</button></p>
      <div className="demo-hint"><strong>Demo login</strong><br />student@college.edu · student123</div>
    </form></div>
  </main>;
}

function ChatWidget({ onOpenEvent }) {
  const [open, setOpen] = useState(false);
  const [messages, setMessages] = useState([{ role: "assistant", text: "Hi! Ask me about event dates, venues, fees or what you should attend." }]);
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const send = async (event) => {
    event.preventDefault();
    const message = draft.trim();
    if (!message || busy) return;
    setDraft(""); setMessages((current) => [...current, { role: "user", text: message }]); setBusy(true);
    try {
      const result = await api("/chat", { method: "POST", body: jsonBody({ message, history: [] }) });
      setMessages((current) => [...current, { role: "assistant", text: result.answer, citations: result.citations || [] }]);
    } catch (error) {
      setMessages((current) => [...current, { role: "assistant", text: error.message }]);
    } finally { setBusy(false); }
  };
  const renderText = (text) => text.split(/(\[E\d+\])/g).map((part, index) => {
    const match = part.match(/^\[E(\d+)\]$/);
    return match ? <button key={index} className="citation-link" onClick={() => onOpenEvent(Number(match[1]), "events")}>{part}</button> : part;
  });
  return <>
    {open && <section className="chat-panel" aria-label="Event assistant">
      <header className="chat-header"><div className="chat-avatar">✦</div><div><strong>Campus guide</strong><span>Event assistant · ready to help</span></div><button className="icon-button" onClick={() => setOpen(false)} aria-label="Close chat">×</button></header>
      <div className="chat-messages">{messages.map((item, index) => <div key={index} className={`chat-message ${item.role}`}><div>{renderText(item.text)}</div></div>)}{busy && <div className="chat-message assistant"><div className="typing">Thinking<span> ···</span></div></div>}</div>
      <form className="chat-compose" onSubmit={send}><input value={draft} onChange={(event) => setDraft(event.target.value)} placeholder="Ask about an event…" aria-label="Message" /><button disabled={busy || !draft.trim()} aria-label="Send message">↑</button></form>
    </section>}
    <button className={`chat-launch ${open ? "chat-launch-open" : ""}`} onClick={() => setOpen(!open)} aria-label={open ? "Close chat" : "Open event assistant"}>{open ? "×" : <><span>✦</span><b>Ask evently</b></>}</button>
  </>;
}

function EventDetail({ event, onBack, onAction, notice }) {
  if (!event) return <div className="empty-state">Loading event…</div>;
  const registrationClosed = new Date(event.registration_deadline) < new Date() || new Date(event.start_time) < new Date();
  return <section className="detail-page">
    <button className="back-link" onClick={onBack}>← Back to events</button>
    <div className="detail-hero"><div><span className="category-pill">{event.category}</span><h1>{event.title}</h1><p>{event.description}</p></div><div className="detail-price">{moneyText(event.fee)}<small>per attendee</small></div></div>
    {notice && <div className="alert alert-success">{notice}</div>}
    <div className="detail-grid"><div className="detail-main">
      <div className="detail-block"><h3>About this event</h3><p>{event.description}</p></div>
      <div className="detail-block"><h3>Good to know</h3><p>{event.rules || "Check the event details with the organizer."}</p><p><strong>Who can attend?</strong> {event.eligibility}</p></div>
      {event.faq?.length > 0 && <div className="detail-block"><h3>Frequently asked</h3>{event.faq.map((item, index) => <details key={index}><summary>{item.q}</summary><p>{item.a}</p></details>)}</div>}
    </div><aside className="detail-aside">
      <div className="detail-info"><span>DATE & TIME</span><strong>{dateText(event.start_time)}</strong></div>
      <div className="detail-info"><span>VENUE</span><strong>{event.venue}</strong></div>
      <div className="detail-info"><span>ORGANIZED BY</span><strong>{event.organizer}</strong></div>
      <div className="detail-info"><span>REGISTRATION CLOSES</span><strong>{dateText(event.registration_deadline)}</strong></div>
      <button className="button button-primary button-wide" disabled={registrationClosed} onClick={() => onAction("register")}>{registrationClosed ? "Registration closed" : "Register for event"} {!registrationClosed && <span>→</span>}</button>
      <button className="button button-outline button-wide" onClick={() => onAction("like")}>♡ Save to my interests</button>
    </aside></div>
  </section>;
}

function AdminPanel({ notify }) {
  const [analytics, setAnalytics] = useState(null);
  const [events, setEvents] = useState([]);
  const [chosenId, setChosenId] = useState("");
  const [registrations, setRegistrations] = useState([]);
  const [busy, setBusy] = useState(false);
  const [form, setForm] = useState({ title: "", description: "", category: CATEGORIES[0], department: DEPARTMENTS[0], organizer: "", venue: "", start_time: "", end_time: "", registration_deadline: "", fee: 0, capacity: 100, eligibility: "Open to enrolled students", rules: "Bring your student ID and arrive 15 minutes early.", faq: [] });
  useEffect(() => {
    Promise.all([api("/admin/analytics"), api("/events?limit=250")]).then(([data, eventRows]) => { setAnalytics(data); setEvents(eventRows); })
      .catch((error) => notify(error.message, "error"));
  }, [notify]);
  useEffect(() => { if (chosenId) api(`/admin/events/${chosenId}/registrations`).then(setRegistrations).catch((error) => notify(error.message, "error")); }, [chosenId, notify]);
  const change = (key) => (event) => setForm((current) => ({ ...current, [key]: event.target.type === "number" ? Number(event.target.value) : event.target.value }));
  const create = async (event) => {
    event.preventDefault(); setBusy(true);
    try {
      const created = await api("/events", { method: "POST", body: jsonBody(form) });
      setEvents((current) => [created, ...current]); setForm((current) => ({ ...current, title: "", description: "", organizer: "", venue: "", start_time: "", end_time: "", registration_deadline: "" }));
      notify("Event published. The assistant has been updated.", "success");
    } catch (error) { notify(error.message, "error"); }
    finally { setBusy(false); }
  };
  return <section className="admin-page">
    <div className="page-heading"><div><p className="eyebrow">CAMPUS OPERATIONS</p><h1>Admin dashboard</h1><p className="muted">Publish events and keep an eye on what your campus is showing up for.</p></div></div>
    {analytics && <div className="stats-grid"><div className="stat-card"><span>Total events</span><strong>{events.length}</strong></div><div className="stat-card"><span>Registrations</span><strong>{analytics.registrations_per_event.reduce((sum, item) => sum + item.registrations, 0)}</strong></div><div className="stat-card"><span>Recommendation impressions</span><strong>{analytics.recommendation_click_through.impressions}</strong></div><div className="stat-card"><span>Click-through rate</span><strong>{analytics.recommendation_click_through.rate == null ? "—" : `${Math.round(analytics.recommendation_click_through.rate * 100)}%`}</strong></div></div>}
    <div className="admin-grid"><form className="panel admin-form" onSubmit={create}><div className="panel-heading"><div><p className="eyebrow">NEW LISTING</p><h2>Publish an event</h2></div><span className="heading-mark">＋</span></div>
      <label>Event title<input required value={form.title} onChange={change("title")} placeholder="e.g. Design Futures Workshop" /></label>
      <label>Description<textarea required rows="3" value={form.description} onChange={change("description")} /></label>
      <div className="form-row"><label>Category<select value={form.category} onChange={change("category")}>{CATEGORIES.map((item) => <option key={item}>{item}</option>)}</select></label><label>Department<select value={form.department} onChange={change("department")}>{DEPARTMENTS.map((item) => <option key={item}>{item}</option>)}</select></label></div>
      <div className="form-row"><label>Organizer<input required value={form.organizer} onChange={change("organizer")} /></label><label>Venue<input required value={form.venue} onChange={change("venue")} /></label></div>
      <div className="form-row"><label>Starts<input required type="datetime-local" value={form.start_time} onChange={change("start_time")} /></label><label>Ends<input required type="datetime-local" value={form.end_time} onChange={change("end_time")} /></label></div>
      <div className="form-row"><label>Registration deadline<input required type="datetime-local" value={form.registration_deadline} onChange={change("registration_deadline")} /></label><label>Fee (₹)<input required type="number" min="0" step="1" value={form.fee} onChange={change("fee")} /></label></div>
      <label>Capacity<input required type="number" min="1" value={form.capacity} onChange={change("capacity")} /></label>
      <label>Eligibility<textarea required rows="2" value={form.eligibility} onChange={change("eligibility")} /></label>
      <label>Rules<textarea required rows="3" value={form.rules} onChange={change("rules")} /></label>
      <button className="button button-primary" disabled={busy}>{busy ? "Publishing…" : "Publish event"}<span>→</span></button>
    </form>
    <div className="panel analytics-panel"><div className="panel-heading"><div><p className="eyebrow">WHAT'S HAPPENING</p><h2>Campus pulse</h2></div><span className="pulse-dot"/></div>
      <h3 className="subheading">Top events by registration</h3>{analytics?.top_events?.length ? <div className="metric-list">{analytics.top_events.slice(0, 6).map((item, index) => <div className="metric-row" key={item.event_id}><span className="metric-index">0{index + 1}</span><span className="metric-title">{item.title}</span><strong>{item.registrations}</strong></div>)}</div> : <p className="muted">No registrations yet.</p>}
      <h3 className="subheading">Likes by category</h3>{analytics?.likes_per_category?.length ? <div className="metric-list">{analytics.likes_per_category.slice(0, 5).map((item) => <div className="metric-row" key={item.category}><span className="metric-title">{item.category}</span><strong>{item.likes}</strong></div>)}</div> : <p className="muted">Category activity will appear here.</p>}
      <label className="registrations-select">View registrations<select value={chosenId} onChange={(event) => setChosenId(event.target.value)}><option value="">Choose an event</option>{events.map((item) => <option key={item.id} value={item.id}>{item.title}</option>)}</select></label>
      {chosenId && <div className="registration-list">{registrations.length ? registrations.map((student) => <div key={student.id}><span className="avatar tiny">{initials(student.name)}</span><span>{student.name}<small>{student.department} · Year {student.year}</small></span></div>) : <p className="muted">No registrations for this event yet.</p>}</div>}
    </div></div>
  </section>;
}

export default function App() {
  const [user, setUser] = useState(null);
  const [authLoading, setAuthLoading] = useState(Boolean(getToken()));
  const [page, setPage] = useState("feed");
  const [selected, setSelected] = useState(null);
  const [recommendations, setRecommendations] = useState([]);
  const [events, setEvents] = useState([]);
  const [notifications, setNotifications] = useState([]);
  const [filters, setFilters] = useState({ q: "", category: "", department: "", free: "", upcoming: true });
  const [notice, setNotice] = useState(null);
  const [busy, setBusy] = useState(false);
  const [authError, setAuthError] = useState("");
  const [interestDraft, setInterestDraft] = useState([]);
  const notify = useCallback((message, kind = "success") => { setNotice({ message, kind }); window.setTimeout(() => setNotice(null), 4200); }, []);
  const go = useCallback((nextPage) => { setPage(nextPage); setSelected(null); window.scrollTo({ top: 0, behavior: "smooth" }); }, []);

  const loadUser = useCallback(async () => {
    if (!getToken()) { setAuthLoading(false); return; }
    try { const profile = await api("/auth/me"); setUser(profile); setInterestDraft(profile.interests || []); }
    catch { setToken(null); setUser(null); }
    finally { setAuthLoading(false); }
  }, []);
  useEffect(() => { loadUser(); }, [loadUser]);

  const loadHome = useCallback(async () => {
    if (!user) return;
    try {
      const [recs, notes] = await Promise.all([api("/recommend?k=8"), api("/notifications?unread_only=true&limit=5")]);
      setRecommendations(recs); setNotifications(notes);
    } catch (error) { notify(error.message, "error"); }
  }, [user, notify]);
  useEffect(() => { if (page === "feed") loadHome(); }, [page, loadHome]);

  const searchEvents = useCallback(async (event) => {
    event?.preventDefault();
    const params = new URLSearchParams();
    if (filters.q.trim()) params.set("q", filters.q.trim());
    if (filters.category) params.set("category", filters.category);
    if (filters.department) params.set("department", filters.department);
    if (filters.free) params.set("free", filters.free);
    if (filters.upcoming) params.set("upcoming", "true");
    try { setEvents(await api(`/events?${params.toString()}`)); }
    catch (error) { notify(error.message, "error"); }
  }, [filters, notify]);
  useEffect(() => { if (page === "events") searchEvents(); }, [page, searchEvents]);

  const login = async (form) => {
    setBusy(true); setAuthError("");
    try {
      const token = await api("/auth/login", { method: "POST", body: jsonBody({ email: form.email, password: form.password }) });
      setToken(token.access_token); const profile = await api("/auth/me"); setUser(profile); setInterestDraft(profile.interests || []);
      go(profile.interests?.length ? "feed" : "onboarding");
    } catch (error) { setAuthError(error.message); }
    finally { setBusy(false); }
  };
  const register = async (form) => {
    setBusy(true); setAuthError("");
    try {
      await api("/auth/register", { method: "POST", body: jsonBody({ ...form, year: Number(form.year), interests: [] }) });
      await login(form);
    } catch (error) { setAuthError(error.message); setBusy(false); }
  };
  const logout = () => { setToken(null); setUser(null); setSelected(null); go("feed"); };
  const saveInterests = async () => {
    try { const profile = await api("/auth/me/interests", { method: "PUT", body: jsonBody({ interests: interestDraft }) }); setUser(profile); go("feed"); }
    catch (error) { notify(error.message, "error"); }
  };
  const openEvent = async (id, source = "events") => {
    try {
      if (user) await api(`/events/${id}/view?source=${source}`, { method: "POST" });
      const event = await api(`/events/${id}`); setSelected(event); setPage("detail"); window.scrollTo({ top: 0, behavior: "smooth" });
    } catch (error) { notify(error.message, "error"); }
  };
  const eventAction = async (eventId, action) => {
    try {
      const result = await api(`/events/${eventId}/${action}`, { method: "POST" });
      notify(result.message || (action === "like" ? "Saved to your interests." : "You're on the list."));
      if (action === "register" || action === "like") loadHome();
    } catch (error) { notify(error.message, "error"); }
  };
  const markRead = async (id) => {
    try { await api(`/notifications/${id}/read`, { method: "POST" }); setNotifications((current) => current.filter((item) => item.id !== id)); }
    catch (error) { notify(error.message, "error"); }
  };
  const nav = useMemo(() => [
    { id: "feed", label: "For you", icon: "✦" }, { id: "events", label: "Explore events", icon: "◌" },
    { id: "notifications", label: "Notifications", icon: "♧", badge: notifications.length },
  ], [notifications.length]);

  if (authLoading) return <div className="loading-screen"><span className="loading-mark">e</span><p>Getting your campus ready…</p></div>;
  if (!user) return <AuthScreen onLogin={login} onRegister={register} busy={busy} error={authError} />;

  return <div className="app-shell">
    <aside className="sidebar"><button className="brand" onClick={() => go("feed")}><span className="brand-mark">e</span><span>evently</span></button>
      <div className="sidebar-label">YOUR CAMPUS</div><nav>{nav.map((item) => <button key={item.id} className={`nav-item ${page === item.id ? "active" : ""}`} onClick={() => go(item.id)}><span className="nav-icon">{item.icon}</span><span>{item.label}</span>{item.badge > 0 && <i className="nav-badge">{item.badge}</i>}</button>)}
        {user.role === "admin" && <><div className="sidebar-label nav-admin-label">MANAGE</div><button className={`nav-item ${page === "admin" ? "active" : ""}`} onClick={() => go("admin")}><span className="nav-icon">⌘</span><span>Admin dashboard</span></button></>}
      </nav>
      <div className="sidebar-bottom"><div className="sidebar-tip"><span>✦</span><strong>A good day starts here.</strong><small>Your next favourite thing is one event away.</small></div><div className="user-menu"><div className="avatar">{initials(user.name)}</div><div className="user-info"><strong>{user.name}</strong><small>{user.role === "admin" ? "Campus admin" : `${user.department} · Year ${user.year}`}</small></div><button className="icon-button logout-button" title="Sign out" onClick={logout}>↗</button></div></div>
    </aside>
    <main className="main-content">
      <header className="topbar"><div className="breadcrumb"><span>evently</span><b>/</b><strong>{page === "feed" ? "For you" : page === "events" ? "Explore events" : page === "detail" ? "Event details" : page === "notifications" ? "Notifications" : page === "onboarding" ? "Your interests" : "Admin dashboard"}</strong></div><div className="topbar-right"><button className="topbar-date">◷ &nbsp;{new Intl.DateTimeFormat("en-IN", { weekday: "short", day: "numeric", month: "short" }).format(new Date())}</button><button className="avatar top-avatar" onClick={() => go("notifications")} title="Notifications">{initials(user.name)}{notifications.length > 0 && <i/>}</button></div></header>
      {notice && <div className={`toast alert-${notice.kind}`}>{notice.kind === "error" ? "!" : "✓"} &nbsp; {notice.message}</div>}
      <div className="content-wrap">
        {page === "onboarding" && <section className="onboarding-page"><p className="eyebrow">A LITTLE ABOUT YOU</p><h1>What are you<br />curious about?</h1><p className="muted">Pick a few interests. We’ll use them to make your event feed feel like yours.</p><div className="interest-chips">{CATEGORIES.map((category) => <button key={category} className={`interest-chip ${interestDraft.includes(category) ? "selected" : ""}`} onClick={() => setInterestDraft((items) => items.includes(category) ? items.filter((item) => item !== category) : [...items, category])}>{interestDraft.includes(category) ? "✓" : "+"} {category}</button>)}</div><button className="button button-primary" disabled={!interestDraft.length} onClick={saveInterests}>Build my feed <span>→</span></button></section>}

        {page === "feed" && <section className="feed-page"><div className="welcome-hero"><div><p className="eyebrow">{new Intl.DateTimeFormat("en-IN", { weekday: "long", month: "long", day: "numeric" }).format(new Date()).toUpperCase()} · YOUR CAMPUS</p><h1>Good things are<br /><em>happening.</em></h1><p>Your next idea, new friend or favourite memory could be on this list.</p><button className="button button-light" onClick={() => go("events")}>Explore all events <span>→</span></button></div><div className="hero-art"><div className="hero-ring ring-one"/><div className="hero-ring ring-two"/><div className="hero-orbit-dot dot-a">✦</div><div className="hero-orbit-dot dot-b">✳</div><div className="hero-sun">e</div><div className="hero-caption">MAKE ROOM<br />FOR SOMETHING NEW</div></div><div className="hero-index">01 <span>/ 08</span></div></div>
          <div className="section-heading"><div><p className="eyebrow">PICKED FOR YOUR INTERESTS</p><h2>Your next favourite</h2></div><button className="text-link" onClick={() => go("events")}>See all events <span>→</span></button></div>
          {recommendations.length ? <div className="event-grid">{recommendations.map((item) => <EventCard key={item.event.id} event={item.event} reason={item.reason} onOpen={() => openEvent(item.event.id, "recommend")} onRegister={() => eventAction(item.event.id, "register")} onLike={() => eventAction(item.event.id, "like")} />)}</div> : <div className="empty-state"><div className="empty-icon">✦</div><h3>No upcoming picks just yet</h3><p>Explore the event list and we’ll get to know what you like.</p><button className="text-link" onClick={() => go("events")}>Explore events →</button></div>}
          <div className="feed-lower"><div className="note-card"><span className="note-flower">✿</span><div><p className="eyebrow">A SMALL NUDGE</p><h3>Try something outside your usual.</h3><p>One new room can lead to a very good story.</p></div><button onClick={() => go("events")}>Browse the calendar →</button></div><div className="mini-notes"><span>✦</span><strong>{notifications.length ? `${notifications.length} updates for you` : "Your campus, in one place"}</strong><p>{notifications.length ? "New events and reminders are waiting." : "Events that match your interests, all together."}</p><button className="text-link" onClick={() => go("notifications")}>Open notifications →</button></div></div>
        </section>}

        {page === "events" && <section className="events-page"><div className="page-heading"><div><p className="eyebrow">THE CAMPUS CALENDAR</p><h1>Find your people.</h1><p className="muted">A workshop, a match, a moment. It’s all happening here.</p></div><div className="heading-deco">✳</div></div>
          <form className="filter-bar" onSubmit={searchEvents}><div className="search-input"><span>⌕</span><input value={filters.q} onChange={(event) => setFilters((current) => ({ ...current, q: event.target.value }))} placeholder="Search events, clubs, ideas…" /></div><select aria-label="Category" value={filters.category} onChange={(event) => setFilters((current) => ({ ...current, category: event.target.value }))}><option value="">All categories</option>{CATEGORIES.map((item) => <option key={item}>{item}</option>)}</select><select aria-label="Department" value={filters.department} onChange={(event) => setFilters((current) => ({ ...current, department: event.target.value }))}><option value="">All departments</option>{DEPARTMENTS.map((item) => <option key={item}>{item}</option>)}</select><select aria-label="Fee" value={filters.free} onChange={(event) => setFilters((current) => ({ ...current, free: event.target.value }))}><option value="">Any fee</option><option value="true">Free</option><option value="false">Paid</option></select><button className="button button-primary" type="submit">Search</button></form>
          <div className="results-line"><span>{events.length} event{events.length === 1 ? "" : "s"} to explore</span><label><input type="checkbox" checked={filters.upcoming} onChange={(event) => setFilters((current) => ({ ...current, upcoming: event.target.checked }))} onClick={(event) => event.stopPropagation()} /> Upcoming only</label></div>
          {events.length ? <div className="event-grid">{events.map((event) => <EventCard key={event.id} event={event} onOpen={() => openEvent(event.id)} onRegister={() => eventAction(event.id, "register")} onLike={() => eventAction(event.id, "like")} />)}</div> : <div className="empty-state"><div className="empty-icon">◌</div><h3>No events match those filters</h3><p>Try a different category or clear your search.</p><button className="text-link" onClick={() => { setFilters({ q: "", category: "", department: "", free: "", upcoming: true }); }}>Clear filters →</button></div>}
        </section>}

        {page === "detail" && <EventDetail event={selected} onBack={() => go("events")} onAction={(action) => eventAction(selected.id, action)} notice={null} />}
        {page === "notifications" && <section className="notifications-page"><div className="page-heading"><div><p className="eyebrow">THE LATEST FROM CAMPUS</p><h1>Your updates.</h1><p className="muted">A heads-up when something good is happening.</p></div></div><div className="notification-list">{notifications.length ? notifications.map((item) => <article className="notification-card" key={item.id}><div className="notification-symbol">✦</div><div><strong>{item.message}</strong><small>{dateText(item.created_at)}</small>{item.event_id && <button className="text-link" onClick={() => openEvent(item.event_id)}>View event →</button>}</div><button className="icon-button mark-read" title="Mark as read" onClick={() => markRead(item.id)}>✓</button></article>) : <div className="empty-state"><div className="empty-icon">♧</div><h3>You’re all caught up</h3><p>We’ll let you know when there’s something worth a look.</p></div>}</div></section>}
        {page === "admin" && user.role === "admin" && <AdminPanel notify={notify} />}
      </div>
      <footer className="page-footer"><span>evently</span><span>Find a little more campus.</span><span>MADE FOR YOUR CAMPUS ✦</span></footer>
    </main>
    <ChatWidget onOpenEvent={openEvent} />
  </div>;
}
