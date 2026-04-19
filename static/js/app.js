// ── Flash auto-dismiss ────────────────────────────────────────────────────
document.querySelectorAll('.flash').forEach(el => {
  setTimeout(() => {
    el.style.transition = 'opacity 0.5s';
    el.style.opacity    = '0';
    setTimeout(() => el.remove(), 500);
  }, 4000);
});

// ── Bar chart tooltip ─────────────────────────────────────────────────────
document.querySelectorAll('.bar').forEach(bar => {
  bar.title = bar.dataset.val || '';
});

// ── Confirm deletes (fallback) ────────────────────────────────────────────
document.querySelectorAll('form[onsubmit]').forEach(form => {
  form.addEventListener('submit', e => {
    if (!confirm('Delete this workout?')) e.preventDefault();
  });
});

// ── Set date input max to today ───────────────────────────────────────────
const dateInput = document.getElementById('workout_date');
if (dateInput) {
  const today = new Date().toISOString().split('T')[0];
  dateInput.setAttribute('max', today);
}
