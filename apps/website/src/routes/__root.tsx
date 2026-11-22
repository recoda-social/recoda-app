import { Outlet, createRootRoute } from "@tanstack/react-router";
import { TanStackRouterDevtoolsPanel } from "@tanstack/react-router-devtools";
import { TanStackDevtools } from "@tanstack/react-devtools";
import { AuthProvider } from "#/auth/AuthProvider";
import { PreviewPlayerProvider } from "#/songs/preview-player";

export const Route = createRootRoute({
  component: RootComponent,
});

function RootComponent() {
  return (
    <PreviewPlayerProvider>
      <AuthProvider>
        <Outlet />
        <TanStackDevtools
          config={{
            position: "bottom-right",
          }}
          plugins={[
            {
              name: "TanStack Router",
              render: <TanStackRouterDevtoolsPanel />,
            },
          ]}
        />
      </AuthProvider>
    </PreviewPlayerProvider>
  );
}
