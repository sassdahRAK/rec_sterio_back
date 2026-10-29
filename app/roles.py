"""The account-role system, in exactly one place.

Every role an `admin_users` row may carry is defined here, and every role
*check* in the application imports its tuple from here. A role never travels
as a bare string outside this module (the database migration in `db.py` and
the tests are the only exceptions), so adding a role means editing one file
instead of hunting literals through the routers.

Roles are **database-authoritative**: a token's `role` claim is informational
only (`auth._resolve_principal` re-reads the row), which is what makes the
role changes of the user-management router take effect on the target's very
next request without re-issuing a token.
"""
from __future__ import annotations

ROLE_SUPER_ADMIN = "super_admin"
ROLE_ADMIN = "admin"
ROLE_DEVELOPER = "developer"
ROLE_REVIEWER = "reviewer"
ROLE_USER = "user"

# Every role an admin_users row may carry. `Principal.is_signed_in` is exactly
# this tuple: a signed-in account is any account, whatever it may do.
ACCOUNT_ROLES = (ROLE_SUPER_ADMIN, ROLE_ADMIN, ROLE_DEVELOPER, ROLE_REVIEWER, ROLE_USER)

# The tier that runs the deployment. Both roles see every folder, publish and
# flip folders to public visibility, and count as "a console admin" to the
# bot's admin-gated commands. User management is deliberately NOT here —
# that belongs to the super admin alone (`require_super_admin`).
ADMIN_TIER = (ROLE_SUPER_ADMIN, ROLE_ADMIN)

# The operator surface: the job queue, global aggregates, retry and cancel
# (access.require_operator). The developer joins the tier because operating
# the pipeline is the job; a reviewer or a plain user must not read the queue
# that serves every folder.
OPERATOR_ROLES = (ROLE_SUPER_ADMIN, ROLE_ADMIN, ROLE_DEVELOPER)

# Roles that see the whole library in the console listings (`GET /videos`,
# the de-duplication scope) rather than only the folders shared with them.
# The reviewer audits transcripts across the dataset, so they see it all.
LIBRARY_VIEW_ROLES = (ROLE_SUPER_ADMIN, ROLE_ADMIN, ROLE_DEVELOPER, ROLE_REVIEWER)

# Roles that may edit any video's transcript whatever folder it lives in
# (access.require_transcript_editor). Curating a dataset means correcting its
# transcripts; it does NOT extend to moving or deleting the media, which keep
# the per-folder "editor" requirement.
TRANSCRIPT_EDIT_ROLES = (ROLE_SUPER_ADMIN, ROLE_ADMIN, ROLE_REVIEWER)